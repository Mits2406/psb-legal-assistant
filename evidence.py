"""Global source retrieval, question-part coverage and bounded document expansion.

No inferred topic removes documents from retrieval. BM25F is always available;
verified portable vectors add dense candidates through rank-only fusion.
An available API reranker prioritizes fused candidates without deleting evidence.
Source IDs refer to unchanged canonical Chunk text, not generated summaries.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from collections import Counter
import json
import math
import re
import time
from corpus import _ENDINGS, _STOP

from registry import contract_number
from query import MONTHS, temporal_mentions, explicit_contracts_in, infer_topic
from retrieval import Retriever, SearchScope
from llm import APIError, number


def _jsonable(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not str(k).startswith('_')}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    return value


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


# Terminology aliases contain no document IDs, clauses, amounts or legal answers.
_ALIASES = (
    (r'главбух', 'главный бухгалтер'),
    (r'испытательн|испытан', 'испытание при приеме на работу срок испытания'),
    (r'минимальн.{0,25}(?:зарплат|заработ|оплат)|\bмрот\b', 'минимальный размер оплаты труда МРОТ'),
    (r'сверхуроч|переработ', 'сверхурочная работа оплата дополнительное время отдыха'),
    (r'поставок|поставк|поставить', 'поставка товара'),
    (r'оплат|плат[её]ж|платить|вносить|внести', 'оплата срок платежа'),
    (r'срок.{0,20}оплат|когда.{0,25}(?:плат|оплат)', 'оплата производится в течение'),
    (r'арендн.{0,12}плат', 'арендная плата размер'),
    (r'коммунальн|электрич|сч[её]тчик', 'коммунальные услуги электроэнергия приборы учета'),
    (r'согласован|согласовыв|согласу|одобр', 'согласование договора регламент'),
    (r'налогов.{0,20}санкц|налогов.{0,20}ответствен', 'налоговые санкции штраф смягчающие обстоятельства'),
    (r'персональн|обработк.{0,12}данн', 'согласие на обработку персональных данных'),
    (r'архив|скан|оригинал', 'хранение оригиналов договоров архив'),
    (r'(?:кто|кем).{0,45}(?:перв|начина|начать|запуска|заняться)|(?:начина|начать|запуска).{0,35}(?:договор|контракт|оформлен|сделк)', 'инициатор договора начало согласования регламент'),
)


def _unnamed_rule_followup(text):
    """A historical question can omit the subject of the previous USER turn.

    Carry vocabulary only for an explicitly dependent, unnamed rule. A newly
    named tax, contract, article or law remains an independent search subject.
    This does not copy an old date or an assistant's asserted legal amount.
    """
    value = text.lower().replace('ё', 'е')
    if not re.search(r'^\s*а\s+(?:како[йевю]|какую|каков[аоы]?|сколько)\b', value):
        return False
    if infer_topic(value) not in {'law', 'generic'} or explicit_contracts_in(value):
        return False
    if not re.search(r'\b(?:закон\w*|норм[аыуе]\w*|стать[яиею]\w*|редакци\w*)\b', value):
        return False
    if not (temporal_mentions(value) or re.search(r'\b(?:тогда|ранее|прежде)|того времени|прошл\w*\s+год', value)):
        return False
    if re.search(r'\b(?:мрот|ндс|ндфл|нк|тк|гк|коап)\b|налог|штраф|пенси|пособи|'
                 r'персональн|сверхуроч|неустойк|\b(?:статья|статье|статью|ст\.)\s*\d|'
                 r'\d+\s*[-–]\s*фз|№\s*\d|[«»"]', value):
        return False
    return True


class EvidencePlanner:
    MAX_CONTEXT_CHARS = 30_000
    SHORT_DOCUMENT_CHARS = 7_000
    MAX_PARTS = 6
    MAX_DOCS = 8
    MAX_ANCHORS = 14

    def __init__(self, corpus, registry, client):
        self.corpus, self.registry, self.client = corpus, registry, client
        self.structure = Retriever(corpus, client, registry)
        self.by_doc = {key: [] for key in corpus.documents}
        suspicious = {}
        for chunk in corpus.chunks.values():
            if chunk.suspicious:
                suspicious.setdefault(chunk.doc_id, []).append((chunk.start, chunk.end))
        self.safe = {}
        for chunk in corpus.chunks.values():
            doc = corpus.documents.get(chunk.doc_id)
            if (not doc or chunk.suspicious or not chunk.searchable
                    or chunk.content_role in {'metadata', 'editorial'}
                    or doc.text[chunk.start:chunk.end] != chunk.text
                    or any(chunk.start < end and start < chunk.end for start, end in suspicious.get(chunk.doc_id, []))):
                continue
            self.safe[chunk.id] = chunk
            self.by_doc[chunk.doc_id].append(chunk)
        for chunks in self.by_doc.values():
            chunks.sort(key=lambda c: (c.start, c.end, c.id))
        self.safe_ids = frozenset(self.safe)
        self._body_roots = {cid: self._focus_roots(c.text) for cid, c in self.safe.items()}
        self._root_frequency = Counter(root for roots in self._body_roots.values() for root in roots)

    @staticmethod
    def _expanded(text):
        aliases = [phrase for pattern, phrase in _ALIASES if re.search(pattern, text, re.I)]
        return text + ('\nПоисковые термины: ' + '; '.join(aliases) if aliases else '')

    def _parts(self, question, ctx):
        continuation = ctx.get('_clarification_continuation')
        if continuation:
            original, latest = continuation['original_task'], continuation['latest_user']
            # Each independent current question keeps the original subject;
            # the old user task is background, never a standalone extra task.
            pieces = [latest, *re.split(r'[;?\n]+|\s+и\s+(?=(?:как|когда|какая|какой|кто|что|сколько|нужно|можно|до\s|за\s))|,\s*(?=а\s|как|кто|когда|каков|какой|что|сколько)', latest, flags=re.I)]
            seen, parts = set(), []
            for piece in pieces:
                key = re.sub(r'\s+', ' ', piece.strip(' ,.:;?!')).casefold()
                if key in seen or len(re.findall(r'[а-яa-z]+', key, re.I)) < 2:
                    continue
                seen.add(key)
                parts.append(original + '\n' + piece.strip())
            return parts[:self.MAX_PARTS] or [original + '\n' + latest]
        semantic_queries = ctx.get('_normalization_queries')
        if isinstance(semantic_queries, list) and semantic_queries:
            # These retain the original substantive task and its validated
            # reconstruction. A fact reply is never an independent query.
            parts = list(dict.fromkeys(value.strip() for value in semantic_queries
                                      if isinstance(value, str) and value.strip()))
            if parts:
                return parts[:self.MAX_PARTS]
        text = str(ctx.get('text') or ctx.get('latest') or question).strip()
        if ctx.get('_clarification_task'):
            # A short reply supplies facts for the previous task, not another
            # independent question. Every task part carries those facts;
            # searching bare 'when to pay?' or 'user clarification' alongside
            # a contract ID would otherwise introduce unrelated laws.
            task = str(ctx['_clarification_task']).lower().replace('ё', 'е')
            facts = ' '.join(str(fact).strip(' .!?') for fact in ctx['_clarification_facts']).lower().replace('ё', 'е')
            pieces = re.split(r'[;?\n]+|\s+и\s+|,\s*(?=а\s|как|кто|когда|каков|какой|что|сколько)', task, flags=re.I)
            parts = [text]
            for part in pieces:
                part = part.strip(' ,.:')
                if len(re.findall(r'[а-яa-z0-9]+', part, re.I)) < 2:
                    continue
                grounded = part + ' ' + facts
                if grounded not in parts:
                    parts.append(grounded)
            return parts[:self.MAX_PARTS]
        # Corrections and elliptical follow-ups keep their subject in previous
        # USER turns, even when the latest turn has no exactly parsed date.
        # This text adds retrieval vocabulary only: the original latest turn
        # and date hints are unchanged, and assistant assertions are excluded.
        latest = str(ctx.get('latest') or question).strip()
        turns = [turn for turn in ctx.get('turns', []) if isinstance(turn, str) and turn.strip()]
        dependent = re.search(
            r'^\s*(?:нет\b|(?:поправ|исправ|уточн|ошиб)[а-яё]*\b|а\s+если\b)'
            r'|\b(?:тот\s+же|та\s+же|те\s+же|тех\s+же|том\s+же|тем\s+же|'
            r'по\s+нему|по\s+ней|в\s+этом\s+случае|меняется\s+ответ)\b', latest, re.I)
        if len(turns) > 1 and (dependent or _unnamed_rule_followup(latest)):
            previous = '\n'.join(turns[-3:-1])
            text = latest + '\n' + previous
        pieces = re.split(r'[;?\n]+|\s+и\s+|,\s*(?=а\s|как|кто|когда|каков|какой|что|сколько)', text, flags=re.I)
        pieces = [p.strip(' ,.:') for p in pieces if len(re.findall(r'[а-яa-z0-9]+', p, re.I)) >= 2]
        parts = [text]
        for part in pieces:
            if part and part != text and part not in parts:
                parts.append(part)
        return parts[:self.MAX_PARTS]

    @staticmethod
    def _substantive(chunk):
        if chunk.content_role == 'heading' or len(chunk.text) < 45:
            return False
        # A document preamble is not the useful fallback for its operative terms.
        if chunk.kind == 'company' and chunk.clause == 'текст' and re.match(r'^\s*ДОГОВОР\b', chunk.text, re.I):
            return False
        return True

    def _constraints(self, ctx):
        requested = {contract_number(n) for n in ctx.get('contracts', [])}
        requested.update(contract_number(n) for n in ctx.get('explicit_contracts', []))
        known = {contract_number(n) for p in self.registry.profiles.values() for n in p.contract_numbers}
        allowed_numbers = requested & known
        unknown = sorted(requested - known)
        return requested, allowed_numbers, unknown

    @staticmethod
    def _multiple_temporal_scenarios(ctx):
        """Do not narrow editions using only the first parseable date.

        Inspect the last user message, not old conversation dates: a correction
        in a later turn can still replace the previous event date. Missing years
        are retained as uncertainty; no year is inferred here.
        """
        text = str(ctx.get('latest') or '').lower().replace('ё', 'е')
        full = temporal_mentions(text)
        hints = {(m['from'].isoformat(), m['to'].isoformat()) for m in full}
        known_days = {(m['from'].day, m['from'].month) for m in full if m['from'] == m['to']}
        partial_days = set()
        month_pattern = '|'.join(sorted(MONTHS, key=len, reverse=True))
        for match in re.finditer(r'(?<![а-я\d])(\d{1,2})\s+(' + month_pattern + r')[а-я]*', text):
            if any(m['start'] <= match.start() and match.end() <= m['end'] for m in full):
                continue
            day, month = int(match[1]), MONTHS[match[2]]
            try:
                date(2000, month, day)
            except ValueError:
                continue
            if (day, month) not in known_days:
                partial_days.add((day, month))
        # Numeric dates without a year need a temporal preposition (or a
        # conjunction after another date), so clause labels are not dates.
        for match in re.finditer(r'(?<![\d./])(\d{1,2})[./](\d{1,2})(?![./]\d|\d)', text):
            if any(m['start'] <= match.start() and match.end() <= m['end'] for m in full):
                continue
            before = text[max(0, match.start()-15):match.start()]
            if not (re.search(r'\b(?:на|от|с|до|после|перед|к|по)\s*$', before)
                    or (full and re.search(r'\bи\s*$', before))):
                continue
            day, month = int(match[1]), int(match[2])
            try:
                date(2000, month, day)
            except ValueError:
                continue
            if (day, month) not in known_days:
                partial_days.add((day, month))
        return len(hints) + len(partial_days) > 1

    def _date_state(self, doc_id, ctx):
        # Parsed dates cannot prove whether the user means a boundary, period,
        # shared year or corrected scenario. Keep source editions and their
        # explicit conditions; generation interprets the original user wording.
        return None

    def _eligible(self, chunk, requested, allowed_numbers, ctx):
        profile = self.registry.profiles.get(chunk.doc_id)
        if requested and profile and profile.document_type in {'contract', 'amendment'}:
            if not allowed_numbers.intersection(contract_number(n) for n in profile.contract_numbers):
                return False
        return True

    @staticmethod
    def _focus_score(query, chunk):
        """Soft relevance signal: payment timing differs from late-payment penalties.

        Candidates remain in the global union; this only orders them. The
        original source heading/body supplies every signal, not document IDs.
        """
        score = chunk.score or 0.0
        if re.search(r'оплат|плат[её]ж|платить|вносить', query, re.I):
            if re.search(r'расч[её]т|порядок.{0,15}оплат', chunk.section, re.I):
                score *= 1.5
                if (re.search(r'срок|когда|числа', query, re.I)
                        and re.search(r'в течение|не позднее|срок|не позже', chunk.text, re.I)):
                    score *= 1.5
            body = re.sub(r'^\s*\d+(?:\.\d+)*[.)]\s*', '', chunk.text)
            if re.match(r'оплата\b|плат[её]ж\b', body, re.I):
                score *= 1.3
            if (re.search(r'просроч|неустойк|штраф|пен[ия]', chunk.text, re.I)
                    and not re.search(r'просроч|неустойк|штраф|пен[ия]', query, re.I)):
                score *= .65
        return score

    def _global_search(self, text):
        # Every safe source is searched, regardless of inferred query topic/code.
        return [c for c in self.corpus.search(text, k=60, chunk_ids=self.safe_ids)
                if c.id in self.safe]

    @staticmethod
    def _focus_roots(text):
        # The follow-up supplies the current task; prior user wording supplies
        # its subject. Short inflections matter too: акт / акту / актом.
        generic = {'просто', 'обязательно', 'подскажите', 'пожалуйста', 'договор',
                   'нужен', 'нужна', 'нужны', 'нужно', 'такой', 'какой', 'сразу',
                   'есть', 'быть', 'срок', 'время', 'теперь', 'потом', 'отнести', 'сделать'}
        roots = set()
        for word in re.findall(r'[а-яё]{3,}', text.lower().replace('ё', 'е')):
            if word in _STOP or word in generic:
                continue
            root = _ENDINGS.sub('', word) if len(word) >= 4 else word
            if len(root) >= 3:
                roots.add(root[:5])
        return roots

    def _continued_score(self, latest, chunk, peak):
        roots = self._focus_roots(latest)
        weights = {root: math.log(1 + len(self.safe) / (1 + self._root_frequency[root]))
                   for root in roots if self._root_frequency[root]}
        body = self._body_roots[chunk.id]
        overlap = sum(weight for root, weight in weights.items() if root in body) / max(1, sum(weights.values()))
        # Preserve contextual BM25 relevance, with a bounded preference for
        # provisions answering the current request rather than its background.
        return (chunk.score or 0) + peak * 1.5 * overlap

    def _retrieve(self, parts, ctx, deadline):
        requested, allowed_numbers, unknown = self._constraints(ctx)
        fused, part_hits, trace = {}, [], []
        state = {'warnings': [], 'api_calls': 0, 'dense_used': False, 'lexical_leaders': [], 'dense_leaders': [], 'focused_lexical_leaders': []}
        continuation = ctx.get('_clarification_continuation') or {}
        latest_focus = continuation.get('latest_user')
        eligible_ids = frozenset(cid for cid,c in self.safe.items()
                                 if self._eligible(c, requested, allowed_numbers, ctx))
        for part in parts:
            grounded_part = part
            if part != parts[0] and re.search(r'\b(?:ее|её|его|их|это|этого|эту|там)\b', part, re.I):
                grounded_part += ' ' + parts[0]
            expanded = self._expanded(grounded_part)
            variants = [(part, 1.0 if latest_focus else .35 if expanded != part else 1.0)]
            if expanded != part:
                variants.append((expanded, .25 if latest_focus else 1.0))
            primary, part_scores = [], {}
            for query, weight in variants:
                hits = self._global_search(query)
                raw_peak = max((c.score or 0 for c in hits), default=0)
                current_part = part.split('\n', 1)[-1] if latest_focus else None
                score_chunk = (lambda c: self._continued_score(current_part, c, raw_peak)) if latest_focus else (lambda c: self._focus_score(query,c))
                hits.sort(key=lambda c: (-score_chunk(c),c.id))
                peak = score_chunk(hits[0]) if hits else 0
                relevant = [c for c in hits if score_chunk(c) >= (peak or 0) * .30
                            and c.id in eligible_ids]
                for rank, chunk in enumerate(relevant, 1):
                    score = weight / (60 + rank)
                    fused[chunk.id] = fused.get(chunk.id, 0.0) + score
                    part_scores[chunk.id] = part_scores.get(chunk.id, 0.0) + score
                # Literal and expanded queries both remain retrieval hypotheses.
                primary.extend(relevant)
                trace.append({'kind': 'lexical', 'query': query, 'candidate_count': len(relevant),
                              'candidate_ids': [c.id for c in relevant], 'global_safe_count': len(self.safe)})
            if latest_focus:
                def operative(cid):
                    chunk = self.safe[cid]
                    contact_block = re.match(r'^\s*\d+[.)]\s*(?:реквизиты|подписи)', chunk.text, re.I)
                    return self._substantive(chunk) and not (contact_block and not re.search(r'реквизит|подписант|банковск|расч[её]тн.{0,12}сч[её]т', latest_focus, re.I))
                primary = [replace(self.safe[cid], score=part_scores[cid]) for cid in
                           sorted({c.id for c in primary}, key=lambda cid: (not operative(cid), -part_scores[cid], cid))]
            else:
                primary = sorted({c.id: c for c in primary}.values(), key=lambda c: (not self._substantive(c), -self._focus_score(expanded,c), c.id))
            lexical_leader = primary[0] if primary else None
            if latest_focus and lexical_leader:
                state['focused_lexical_leaders'].append(lexical_leader.id)
            dense = self.structure.dense_search(grounded_part, scope=SearchScope(chunk_ids=eligible_ids), deadline=deadline)
            cache = dense.trace['cache']
            state['warnings'].extend(dense.warnings)
            state['api_calls'] += cache.get('api_calls', 0)
            trace.append({'kind': 'dense', 'query': grounded_part, 'candidate_count': len(dense.chunks),
                          'candidate_ids': [c.id for c in dense.chunks], 'cache': cache,
                          'timings_ms': dense.trace['timings_ms']})
            if dense.chunks:
                state['dense_used'] = True
                state['dense_leaders'].append(dense.chunks[0].id)
                if lexical_leader:
                    state['lexical_leaders'].append(lexical_leader.id)
                # Rank-only fusion: cosine scores and BM25F scales never mix.
                # Keep every accepted lexical candidate in this union.
                for rank, chunk in enumerate(dense.chunks, 1):
                    if chunk.id not in eligible_ids:
                        continue
                    score = 1.0 / (60 + rank)
                    fused[chunk.id] = fused.get(chunk.id, 0.0) + score
                    part_scores[chunk.id] = part_scores.get(chunk.id, 0.0) + score
                ids = set(c.id for c in primary) | {c.id for c in dense.chunks if c.id in eligible_ids}
                primary = [replace(self.safe[cid], score=part_scores[cid]) for cid in
                           sorted(ids, key=lambda cid: (not self._substantive(self.safe[cid]), -part_scores[cid], cid))]
            part_hits.append(primary)
        ranked = sorted(fused, key=lambda cid: (-fused[cid], not self._substantive(self.safe[cid]), cid))
        return [self.safe[cid] for cid in ranked], part_hits, fused, unknown, trace, state

    def _rerank_candidates(self, parts, ranked, by_part, scores, state, deadline):
        """Prioritize canonical candidates once; leave the full baseline on failure.

        The server's top-N is a promoted prefix, never permission to remove the
        omitted tail or independent question parts. Packing still expands the
        original leaders, related clauses and applicable document families.
        """
        started = time.monotonic()
        model = getattr(self.client, 'reranker_model', '')
        enabled = bool(getattr(self.client, 'reranker_enabled', False))
        audit = {'enabled': enabled, 'model': model if isinstance(model, str) else '',
                 'status': 'disabled', 'request_attempted': False, 'api_attempts': 0,
                 'candidate_ids': [], 'returned_ids': [],
                 'result_ids': [c.id for c in ranked], 'preserved_part_leaders': [],
                 'elapsed_ms': 0}
        state['rerank'] = audit
        if not enabled or not isinstance(model, str) or not model.strip():
            audit['status'] = 'not_configured' if not model else 'disabled'
            return ranked, by_part, scores
        if not ranked:
            audit['status'] = 'skipped_empty'
            return ranked, by_part, scores
        if deadline - started < .6:
            audit['status'] = 'skipped_deadline'
            return ranked, by_part, scores

        baseline_ids = [c.id for c in ranked]
        baseline_set = set(baseline_ids)
        original_leaders = list(dict.fromkeys(hits[0].id for hits in by_part if hits))
        required = list(dict.fromkeys(original_leaders + state['lexical_leaders'] + state['dense_leaders']))
        cap = max(int(number('RERANK_CANDIDATES', 32, 8, 64)), len(required))
        chosen = set((required + [cid for cid in baseline_ids if cid not in required])[:cap])
        candidate_ids = [cid for cid in baseline_ids if cid in chosen]
        candidates = [self.safe[cid] for cid in candidate_ids]
        audit['candidate_ids'] = candidate_ids
        call_deadline = min(deadline - .25, started + number('RERANK_TIMEOUT_SECONDS', 8, .5, 10))
        audit['budget_ms'] = round((call_deadline - started) * 1000)
        audit['request_attempted'] = True
        try:
            returned = self.client.rerank(parts[0], candidates, call_deadline)
            if time.monotonic() > call_deadline:
                raise APIError('Исчерпан бюджет реранжирования.', code='rerank_deadline')
            if not isinstance(returned, (list, tuple)) or not returned or len(returned) > len(candidates):
                raise APIError('Недопустимый список реранкера.', code='invalid_rerank_candidates')
            returned_ids = [item.id for item in returned]
            if len(set(returned_ids)) != len(returned_ids) or not set(returned_ids) <= set(candidate_ids):
                raise APIError('Реранкер изменил набор источников.', code='invalid_rerank_candidates')
            for item in returned:
                canonical = self.safe[item.id]
                if any(getattr(item, field, None) != getattr(canonical, field, None)
                       for field in ('doc_id', 'title', 'clause', 'section', 'text', 'start', 'end')):
                    raise APIError('Реранкер изменил источник.', code='invalid_rerank_candidates')
            # Never mix old BM25 magnitudes with the new ordering: both global
            # and per-part consumers receive the same ordinal priority scale.
            promoted = set(returned_ids)
            ordered = returned_ids + [cid for cid in baseline_ids if cid not in promoted]
            positions = {cid: i for i, cid in enumerate(ordered)}
            new_scores = {cid: 1.0 / (60 + i + 1) for cid, i in positions.items()}
            new_parts, allowed_part_documents = [], []
            for hits in by_part:
                part_ids = list(dict.fromkeys(c.id for c in hits if c.id in baseline_set))
                part_set = set(part_ids)
                part_order = [cid for cid in returned_ids if cid in part_set]
                part_order += [cid for cid in part_ids if cid not in promoted]
                new_parts.append([replace(self.safe[cid], score=1.0 / (60 + i + 1))
                                  for i, cid in enumerate(part_order)])
                # Ordinal positions are not calibrated relevance scores.
                # Reusing the old 60% score test on them would admit unrelated
                # tail documents. Preserve the baseline document gate and add
                # only documents explicitly selected by the reranker.
                allowed = {hits[0].doc_id} if hits else set()
                peak = (hits[0].score or 0) if hits else 0
                for chunk in hits[1:]:
                    if chunk.doc_id not in allowed and (chunk.score or 0) >= peak * .60:
                        allowed.add(chunk.doc_id)
                        if len(allowed) >= 3:
                            break
                allowed.update(self.safe[cid].doc_id for cid in returned_ids if cid in part_set)
                allowed_part_documents.append(sorted(allowed))
            audit.update(status='applied', returned_ids=returned_ids, result_ids=ordered,
                         preserved_part_leaders=original_leaders)
            state['pre_rerank_part_leaders'] = original_leaders
            state['rerank_allowed_part_documents'] = allowed_part_documents
            return [self.safe[cid] for cid in ordered], new_parts, new_scores
        except (APIError, ValueError, TypeError, AttributeError, OSError) as exc:
            code = getattr(exc, 'code', 'invalid_rerank_response')
            audit.update(status='failed', error_code=code if isinstance(code, str)
                         and re.fullmatch(r'[a-z0-9_]{1,64}', code) else 'rerank_error')
            state['warnings'].append('Реранжирование недоступно или вернуло некорректный ответ; сохранён исходный порядок поиска.')
            return ranked, by_part, scores
        finally:
            diagnostics = getattr(self.client, 'last_rerank_diagnostics', {})
            if isinstance(diagnostics, dict):
                for key in ('api_attempted', 'api_attempts', 'api_http_success', 'http_status',
                            'top_n', 'returned'):
                    if key in diagnostics:
                        audit[key] = diagnostics[key]
            attempts = audit.get('api_attempts', 0)
            if type(attempts) is int and attempts >= 0:
                state['api_calls'] += attempts
            audit['elapsed_ms'] = round((time.monotonic() - started) * 1000, 2)

    def _families(self, selected, ctx):
        requested, allowed_numbers, _ = self._constraints(ctx)
        profiles = self.registry.profiles
        chosen = set(selected)
        if allowed_numbers:
            chosen.update(doc_id for doc_id, p in profiles.items()
                          if allowed_numbers.intersection(contract_number(n) for n in p.contract_numbers))
        numbers = {contract_number(n) for doc_id in chosen if doc_id in profiles
                   for n in profiles[doc_id].contract_numbers}
        families = {profiles[doc_id].family for doc_id in chosen if doc_id in profiles and profiles[doc_id].family}
        for doc_id, profile in profiles.items():
            same_contract = bool(numbers.intersection(contract_number(n) for n in profile.contract_numbers))
            same_family = bool(profile.family and profile.family in families)
            if same_contract or same_family:
                chosen.add(doc_id)
        return list(selected) + sorted(chosen - set(selected))

    def _protected_documents(self, parts, ranked, ctx):
        """Reserve explicitly requested source types before lexical truncation.

        This adds candidates, never masks the global search. Source names and
        entities are compared generically; linked versions are added afterwards.
        """
        text = self._expanded(parts[0]).lower().replace('ё', 'е')
        patterns = {'regulation': r'регламент', 'letter': r'письм',
                    'amendment': r'дополнительн.{0,15}соглашен|допсоглашен'}
        kinds = {kind for kind, pattern in patterns.items() if re.search(pattern, text)}
        requested, allowed, _ = self._constraints(ctx)
        rank = {c.doc_id: i for i, c in reversed(list(enumerate(ranked)))}
        def words(value):
            return {w[:5] for w in re.findall(r'[а-яa-z]{4,}', value.lower().replace('ё', 'е'))}
        query_words = words(text)
        candidates = []
        for doc_id, profile in self.registry.profiles.items():
            chunks = self.by_doc.get(doc_id, [])
            if profile.document_type not in kinds or not chunks:
                continue
            if not self._eligible(chunks[0], requested, allowed, ctx):
                continue
            identity = ' '.join(profile.entities) + ' ' + self.corpus.documents[doc_id].title
            overlap = len(words(identity) & query_words)
            candidates.append((doc_id, overlap))
        candidates.sort(key=lambda pair: (-pair[1], rank.get(pair[0], len(ranked)), pair[0]))
        return [doc_id for doc_id, _ in candidates[:self.MAX_DOCS]]

    def _card(self, doc_id, ctx):
        doc = self.corpus.documents[doc_id]
        p = self.registry.profiles.get(doc_id)
        card = {'doc_id': doc_id, 'title': doc.title, 'kind': doc.kind}
        if p:
            if p.contract_numbers: card['contracts'] = list(p.contract_numbers)
            if p.family: card['family'] = p.family
            if p.conditions:
                card['conditions'] = [{'field': c.field, 'op': c.op, 'value': c.value} for c in p.conditions]
                card['date_match'] = self._date_state(doc_id, ctx)
            if p.issues: card['unparsed_conditions'] = list(p.issues)[:2]
        amendments = [a for a in self.registry.amendments if a.doc_id == doc_id]
        if amendments:
            card['amendments'] = [{'contract': a.contract_no, 'clause': a.target_clause,
                                   'conditions': [{'field': c.field, 'op': c.op, 'value': c.value} for c in a.conditions]}
                                  for a in amendments]
        if doc.kind == 'law':
            card['snapshot'] = doc.snapshot_date
            card['historical_editions_available'] = False
            # Operative clauses only: editorial notices cannot pretend a body exists.
            card['available_clauses'] = list(dict.fromkeys(c.clause.replace('ст. ' + str(doc.metadata.get('article')), '').strip(' ,') or 'article'
                                                          for c in self.by_doc[doc_id]))
        return card

    def _neighbours(self, anchor):
        same = [c for c in self.by_doc[anchor.doc_id]
                if c.clause == anchor.clause and c.section == anchor.section]
        connected, lo, hi = {anchor.id}, anchor.start, anchor.end
        changed = True
        while changed:
            changed = False
            for c in same:
                if c.id not in connected and c.start < hi and lo < c.end:
                    connected.add(c.id);lo=min(lo,c.start);hi=max(hi,c.end);changed=True
        peers = sorted((self.safe[cid] for cid in connected if cid != anchor.id),
                       key=lambda c: (abs(c.start-anchor.start),c.start))
        parent = self.structure._parent(anchor, self.safe)
        return parent, peers

    def build(self, question: str, ctx: dict, *, deadline: float) -> dict:
        started = time.monotonic()
        ctx = dict(ctx)
        ctx['_multiple_temporal_scenarios'] = self._multiple_temporal_scenarios(ctx)
        parts = self._parts(question, ctx)
        ranked, by_part, scores, unknown, search_trace, retrieval_state = self._retrieve(parts, ctx, deadline)
        ranked, by_part, scores = self._rerank_candidates(parts, ranked, by_part, scores, retrieval_state, deadline)
        protected = self._protected_documents(parts, ranked, ctx)
        protected_family = set(self._families(protected, ctx)) if protected else set()
        anchors, selected, alternatives = [], list(protected), []
        # Reserve the leading document of every question part before adding
        # runners-up, so an early topic cannot consume all document slots.
        for part_index, hits in enumerate(by_part):
            if not hits:
                continue
            anchors.append(hits[0])
            if hits[0].doc_id not in selected:selected.append(hits[0].doc_id)
            peak = hits[0].score or 0
            seen_docs = {hits[0].doc_id}
            rerank_gate = (set(retrieval_state['rerank_allowed_part_documents'][part_index])
                           if retrieval_state['rerank']['status'] == 'applied' else None)
            for chunk in hits[1:]:
                qualifies = (chunk.doc_id in rerank_gate if rerank_gate is not None
                             else (chunk.score or 0) >= peak * .60)
                if chunk.doc_id not in seen_docs and qualifies:
                    alternatives.append(chunk);seen_docs.add(chunk.doc_id)
                    if len(seen_docs) >= 3:break
        # A named source is retained even when no chunk cleared the global
        # lexical threshold. Its safe full text supplies the missing vocabulary.
        for doc_id in protected:
            best = next((c for c in ranked if c.doc_id == doc_id and self._substantive(c)), None)
            if best is None:
                best = next((c for c in self.by_doc[doc_id] if self._substantive(c)), None)
            if best is not None:anchors.append(best)
        # Reserve the strongest lexical and dense evidence for each part beside
        # the fused leader. A useful dense-only source must survive the later
        # relative-score cutoff; fusion must also preserve lexical coverage.
        for cid in dict.fromkeys(retrieval_state.get('pre_rerank_part_leaders', [])
                                 + retrieval_state['lexical_leaders'] + retrieval_state['dense_leaders']):
            chunk = self.safe[cid]
            anchors.append(chunk)
            if chunk.doc_id not in selected:selected.append(chunk.doc_id)
        leading_ids = {c.id for c in anchors}
        for chunk in alternatives:
            if chunk.doc_id not in selected and len(selected) < self.MAX_DOCS:
                selected.append(chunk.doc_id);anchors.append(chunk)
        anchors.extend(c for c in ranked if c.doc_id in selected)
        anchors = list({c.id:self.safe[c.id] for c in anchors}.values())[:max(self.MAX_ANCHORS,len(leading_ids))]
        # Per-part leaders are reserved alongside named documents; only
        # optional alternatives have a document-count limit.
        selected = self._families(selected, ctx)
        selected_set = set(selected)
        anchors = [c for c in anchors if c.doc_id in selected_set]
        full = {d for d in selected if sum(len(c.text) for c in self.by_doc[d]) <= self.SHORT_DOCUMENT_CHARS}
        expected = {c.id for d in full for c in self.by_doc[d]}
        parents, neighbours = [], []
        for anchor in anchors:
            parent, peers = self._neighbours(anchor)
            expected.update(c.id for c in peers)
            expected.add(anchor.id)
            if parent is not None and parent.id in self.safe:
                parents.append(parent);expected.add(parent.id)
            neighbours.extend(peers)
        all_full = [c for doc_id in selected if doc_id in full for c in self.by_doc[doc_id]]
        # Fallback sees topical provisions first, never document-order preambles.
        all_full.sort(key=lambda c: (-scores.get(c.id,0),not self._substantive(c),selected.index(c.doc_id),c.start))
        leading = [c for c in anchors if c.id in leading_ids]
        protected_full = [c for c in all_full if c.doc_id in protected_family]
        candidates = list({c.id:c for c in [*leading,*protected_full,*anchors,*parents,*all_full,*neighbours]}.values())
        hints = _jsonable({key:ctx[key] for key in ('event','event_span','spec_date','spec_span') if ctx.get(key) is not None})
        context = {'parsed_date_hints': hints, 'date_hint_role': 'Unverified parser hints, never edition filters. Interpret temporal scope from the latest user wording and dialogue corrections; source conditions determine applicability.'}
        if ctx.get('comparison') is not None:context['comparison']=ctx['comparison']
        context.update(_jsonable(ctx.get('_answer_context', {})))
        if ctx['_multiple_temporal_scenarios']:
            context.update(event=None, event_span=None, event_date=None, temporal_ambiguity=True,
                           date_hint='Several distinct dates occur in the last user message; no single event date selects an edition. Evaluate each scenario separately or conditionally.')
        cards = [self._card(d,ctx) for d in selected]
        context.update(documents=cards, unknown_contracts=unknown,
                       metadata_role='Document metadata guides applicability; only source text proves legal facts.',
                       evidence_complete=True)
        if not selected:
            context['available_documents'] = [d.title for d in self.corpus.documents.values()]
        sources, used, omitted = [], 0, []
        reserve = 1000
        budget = max(0,self.MAX_CONTEXT_CHARS-_size({'question':question,'context':context})-reserve)
        for chunk in candidates:
            source={'id':f'S{len(sources)+1:03d}','title':chunk.title,'clause':chunk.clause,'text':chunk.text,
                    'doc_id':chunk.doc_id,'chunk_id':chunk.id}
            cost=_size({k:source[k] for k in ('id','title','clause','text')})+1
            if used+cost>budget:
                omitted.append(chunk.id);continue
            sources.append(source);used+=cost
        included={s['chunk_id'] for s in sources}
        missing=expected-included
        for card in cards:
            doc_id=card['doc_id'];count=sum(c.id in included for c in self.by_doc[doc_id])
            card['coverage']='full_safe_document' if count==len(self.by_doc[doc_id]) else 'selected_spans'
            card['omitted_chunks']=len(self.by_doc[doc_id])-count
        context['evidence_complete']=not bool(missing)
        if missing:
            context['incomplete_documents']=list(dict.fromkeys(self.safe[cid].doc_id for cid in sorted(missing)))
        warnings=list(ctx.get('warnings',[])) + retrieval_state['warnings']
        if missing:warnings.append('Часть связанного контекста не вошла в бюджет; не делать вывод о полноте исключений без проверки пропущенных частей.')
        if unknown:warnings.append('Указанный договор отсутствует в доступном каталоге; другие договоры не являются его заменой.')
        by_chunk={s['chunk_id']:s['id'] for s in sources}
        fallback=[]
        fallback_parts = by_part[1:]+by_part[:1]
        if (ctx.get('_clarification_continuation') and retrieval_state['focused_lexical_leaders']
                and retrieval_state['rerank']['status'] != 'applied'):
            # Query-focused literal anchors provide the extraction roots;
            # dense alternatives still remain in the generator's evidence.
            fallback_parts = [[self.safe[cid]] for cid in retrieval_state['focused_lexical_leaders']]
        for hits in fallback_parts:
            for chunk in hits:
                if chunk.id in by_chunk and self._substantive(chunk):
                    if by_chunk[chunk.id] not in fallback:fallback.append(by_chunk[chunk.id])
                    break
        if ctx.get('_clarification_continuation'):
            # Extraction may expand dependencies of these anchors, but a
            # complete document in the LLM context is not another user question.
            context['fallback_anchor_only'] = True
            context['answer_anchor_ids'] = fallback[:8]
        else:
            for source in sources:
                if source['id'] not in fallback and self._substantive(self.safe[source['chunk_id']]):fallback.append(source['id'])
        payload=lambda:{'question':question,'context':context,'sources':[{k:s[k] for k in ('id','title','clause','text')} for s in sources]}
        actual=_size(payload())
        # Reserved metadata space normally makes this unnecessary, but protect
        # the cap if an unusually long condition description is present.
        while sources and actual>self.MAX_CONTEXT_CHARS:
            removed=sources.pop();omitted.append(removed['chunk_id']);context['evidence_complete']=False
            fallback=[sid for sid in fallback if sid!=removed['id']];actual=_size(payload())
        if 'answer_anchor_ids' in context:
            context['answer_anchor_ids'] = fallback[:8]
        trace={'strategy':'global_hybrid_rrf_parts_v15' if retrieval_state['dense_used'] else 'global_lexical_parts_v8',
               'api_calls':retrieval_state['api_calls'],'lexical_leaders_preserved':retrieval_state['lexical_leaders'],
               'dense_leaders_preserved':retrieval_state['dense_leaders'],'temporal_ambiguity':ctx['_multiple_temporal_scenarios'],'query_parts':parts,'searches':search_trace,
               'protected_doc_ids':protected,'selected_doc_ids':selected,'source_count':len(sources),'ranked_chunk_ids':[c.id for c in ranked],
               'budget':{'max_chars':self.MAX_CONTEXT_CHARS,'used_chars':actual,'dropped_chunk_ids':omitted},
               'total_ms':round((time.monotonic()-started)*1000,2)}
        trace['rerank'] = retrieval_state['rerank']
        if retrieval_state['rerank']['status'] == 'applied':
            trace['strategy'] = ('global_hybrid_rrf_rerank_parts_v19' if retrieval_state['dense_used']
                                 else 'global_lexical_rerank_parts_v19')
        return {'sources':sources,'fallback_sources':fallback[:8], 'context':context,'trace':trace,
                'warnings':list(dict.fromkeys(warnings))}
