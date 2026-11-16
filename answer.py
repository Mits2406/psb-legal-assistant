"""One evidence and answer pipeline locally and in the submitted container."""
from __future__ import annotations
import re
import time
from datetime import date
from pathlib import Path
from corpus import Corpus
from registry import RuleRegistry, contract_number
from llm import ModelClient, env_first, load_env, number
from clarification import PreRetrievalClarifier, conversation_context
from evidence import EvidencePlanner
from generation import GroundedGenerator
from answer_style import render_author_answer
from normalization import normalize_query, bind_pending


class LegalAssistant:
    def __init__(self, data_dir=None):
        load_env(Path(__file__).with_name('.env'))
        self.corpus = Corpus(Path(data_dir or env_first('DATA_DIR') or Path(__file__).with_name('data')))
        self.client = ModelClient()
        self.registry = RuleRegistry(self.corpus)
        self.clarifier = PreRetrievalClarifier(self.registry)
        self.planner = EvidencePlanner(self.corpus, self.registry, self.client)

    def _checked_sources(self, sources):
        result, seen = [], set()
        for source in sources:
            chunk = self.corpus.get_chunk(source.get('chunk_id'))
            sid = source.get('id')
            if chunk is None or not isinstance(sid, str) or sid in seen:
                raise ValueError('Invalid evidence identity')
            doc = self.corpus.get_document(chunk.doc_id)
            if (chunk.suspicious or not chunk.searchable or chunk.content_role in {'metadata', 'editorial'}
                    or source.get('text') != chunk.text or source.get('doc_id') != chunk.doc_id
                    or doc.text[chunk.start:chunk.end] != chunk.text):
                raise ValueError('Invalid evidence span')
            seen.add(sid)
            checked = {**source, 'title': doc.title, 'clause': chunk.clause}
            if doc.kind == 'law':
                # Numbered-list labels can continue over later unnumbered law
                # paragraphs. Display the actual article and exact text anchor,
                # without pretending the inherited number identifies that norm.
                prefix = chunk.text.strip()[:80]
                if len(chunk.text.strip()) > 80 and ' ' in prefix:
                    prefix = prefix.rsplit(' ', 1)[0]
                checked['display_label'] = doc.title + ', фрагмент «' + prefix + ('…' if len(prefix) < len(chunk.text.strip()) else '') + '»'
            result.append(checked)
        # Recovery follows prerequisites, not whole documents. A definition or
        # edition header can support many rules without dragging all those rules
        # into an answer about one of them. Model payloads remain unchanged.
        short_company = {s['doc_id'] for s in result
                         if self.corpus.get_document(s['doc_id']).kind == 'company'
                         and len(self.corpus.get_document(s['doc_id']).text) <= 7000}
        def section(source):
            match = re.match(r'^(\d+)(?:\.|$)', str(source.get('clause', '')))
            return match[1] if match else None

        for source in result:
            did = source['doc_id']
            if did not in short_company:
                continue
            profile = self.registry.profiles.get(did)
            own_section = section(source)
            references = set(re.findall(r'\b(?:п\.|пункт\w*)\s*(\d+(?:\.\d+)*)', source['text'], re.I))
            qualifiers = r'не\s+(?:чаще|более|менее)|только|исключени|ограничен|освобожда|по соглашению|при условии|считается исполненным'
            links = []
            for other in result:
                other_profile = self.registry.profiles.get(other['doc_id'])
                same_doc = other['doc_id'] == did
                same_family = bool(profile and other_profile and profile.family
                                   and profile.family == other_profile.family)
                same_clause = source['clause'] == other['clause'] and own_section is not None
                reference = any(str(other.get('clause', '')) == ref
                                or str(other.get('clause', '')).startswith(ref + '.') for ref in references)
                if same_doc and (same_clause or reference):
                    links.append(other['id'])
                elif same_family and (same_clause or reference):
                    links.append(other['id'])
                if same_doc and own_section is not None and section(other) == own_section and re.search(qualifiers, other['text'], re.I):
                    links.append(other['id'])
                if same_family and own_section is not None and re.search(r'только|исключени|вместо|независимо|в дополнение', other['text'], re.I):
                    subject = {word[:5] for word in re.findall(r'[а-яё]{5,}', source['text'].lower())}
                    qualification = {word[:5] for word in re.findall(r'[а-яё]{5,}', other['text'].lower())}
                    shared = (subject & qualification) - {'догов', 'сторо', 'пункт', 'суммы', 'сумма'}
                    # Shared departments and generic procedural verbs do not
                    # prove that two clauses regulate the same object. Keep a
                    # specific subject match before linking across sections.
                    procedural = {'подпи', 'юриди', 'отдел', 'согла', 'оформ', 'перед',
                                  'получ', 'общес', 'насто', 'произ', 'должн',
                                  'течен', 'рабоч', 'кален', 'банко', 'после'}
                    if len(shared) >= 2 and shared - procedural:
                        links.append(other['id'])
                # Definitions of an event date qualify periods counted from it.
                if same_doc:
                    definition = re.search(r'\bдат(?:ой|а)\s+([а-яё -]{3,45}?)\s+(?:считается|признаётся|является)', other['text'], re.I)
                    if definition and re.search(r'\bдат\w*\s+' + re.escape(definition[1]), source['text'], re.I):
                        links.append(other['id'])
                if profile and profile.document_type == 'regulation' and same_doc and other['clause'] == 'текст':
                    if re.search(r'действует|редакци|утратил', other['text'], re.I):
                        links.append(other['id'])
            source['dependency_ids'] = list(dict.fromkeys(
                [*source.get('dependency_ids', []), *(sid for sid in links if sid != source['id'])]))
        # An amendment belongs to its target clause, not to every
        # independent question mentioning the same contract number.
        for amendment in self.registry.amendments:
            amendment_ids = [s['id'] for s in result if s['doc_id'] == amendment.doc_id]
            if not amendment_ids or amendment.doc_id not in short_company:
                continue
            target_ids = []
            for source in result:
                profile = self.registry.profiles.get(source['doc_id'])
                if (source['doc_id'] in short_company and profile and profile.document_type == 'contract'
                        and contract_number(amendment.contract_no) in {contract_number(n) for n in profile.contract_numbers}
                        and (source['clause'] == amendment.target_clause
                             or str(source['clause']).startswith(amendment.target_clause + '.'))):
                    source['dependency_ids'] = list(dict.fromkeys(source['dependency_ids'] + amendment_ids))
                    target_ids.append(source['id'])
            # Either edition may be the lexical anchor. Both branches and the
            # amendment's applicability condition must survive recovery.
            for source in result:
                if source['id'] in amendment_ids:
                    source['dependency_ids'] = list(dict.fromkeys(source.get('dependency_ids', []) + target_ids))
        return result

    def answer(self, question, history=None, event_date=None, deadline=None,
               dialogue_state=None, reference_date=None):
        started = time.monotonic()
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Нужен непустой вопрос.')
        local_deadline = started + number('ANSWER_TIMEOUT_SECONDS', 120, 2, 180)
        deadline = min(deadline, local_deadline) if deadline is not None else local_deadline
        # A clock anchor resolves relative wording; it is never the date of a
        # contract, specification, or corpus snapshot by itself.
        reference_date = reference_date or env_first('REFERENCE_DATE') or date.today().isoformat()
        reference_date = date.fromisoformat(str(reference_date)).isoformat()
        normalized = normalize_query(
            question, history, event_date, reference_date, self.registry,
            self.clarifier, self.client, deadline, state=dialogue_state)
        ctx = normalized['context']
        dialogue_state = normalized['state']
        normalization_trace = normalized['trace']
        clarification = self.clarifier.check(ctx)
        if clarification is not None:
            return {**clarification.response(), 'question': ctx['latest'],
                    'dialogue_state': bind_pending(dialogue_state, clarification),
                    'trace': {'engine': 'v12_semantic', 'api_configured': self.client.enabled,
                              'total_ms': round((time.monotonic() - started) * 1000),
                              'normalization': normalization_trace,
                              'model_calls': normalization_trace.get('calls', 0),
                              'source_count': 0,
                              'clarification': {'stage': 'pre_retrieval', 'reason': clarification.reason},
                              'retrieval': {'skipped': True, 'reason': 'clarification_required', 'api_calls': 0},
                              'generation': {'calls': 0, 'stage': 'pre_retrieval', 'finish_reason': 'not_called',
                                             'api_attempts': 0, 'api_attempted': False,
                                             'api_http_success': False, 'model_text_received': False,
                                             'model_text_usable': False, 'stages': []}}}
        # Dates and entities are hints; applicability follows the actual sources.
        context = {'event_date': event_date,
                   'reference_date': reference_date,
                   'reference_date_role': 'Explicit clock anchor for relative words only; never evidence of a legal rule or the date of a user event.',
                   'date_interpretation': 'Only an explicitly supplied event_date is exact. Interpret relative dates, ranges, multiple events and corrections from the user messages themselves.',
                   'user_turns': ctx['turns'],
                   'contract_hints': ctx['contracts'],
                   'identity_conflict': ctx.get('identity_conflict', False)}
        if ctx.get('semantic'):
            context['query_interpretation'] = ctx['semantic']
            context['query_interpretation_role'] = 'Validated interpretation of user wording, not a source of law. Original messages take precedence; do not drop independent questions or infer missing facts.'
        context['original_question'] = question
        continuation = ctx.get('_clarification_continuation')
        if continuation:
            context['current_question'] = continuation['latest_user']
            context['dialogue_subject'] = continuation['original_task']
        if ctx.get('_original_messages'):
            context['original_messages'] = ctx['_original_messages']
        if ctx.get('resolved_question'):
            context['resolved_question'] = ctx['resolved_question']
        if ctx.get('clarification_reply'):
            context['clarification_reply'] = ctx['clarification_reply']
        if question != ctx['latest'] and ctx.get('history_scope') != 'current_turn':
            context['conversation'] = question
        if history and ctx.get('history_scope') != 'current_turn':
            context['history'] = [h for h in history[-16:] if isinstance(h, dict)
                                  and h.get('role') in {'user', 'assistant'}
                                  and isinstance(h.get('content'), str)]
        ctx['_answer_context'] = context
        if self.planner.client is not self.client:
            self.planner = EvidencePlanner(self.corpus, self.registry, self.client)
        retrieval_budget = number('RETRIEVAL_TIMEOUT_SECONDS', 15, 1, 60)
        retrieval_deadline = min(deadline, time.monotonic() + retrieval_budget)
        plan = self.planner.build(question, ctx, deadline=retrieval_deadline)
        sources = self._checked_sources(plan['sources'])
        generated = GroundedGenerator(self.client).generate(
            ctx['latest'] if continuation else ctx.get('resolved_question', ctx['latest']),
            sources, context=plan.get('context', context), deadline=deadline,
            fallback_ids=plan.get('fallback_ids', plan.get('fallback_sources', [])))
        by_id = {s['id']: s for s in sources}
        citations = []
        for sid in dict.fromkeys(generated.get('used_ids', [])):
            if sid not in by_id:
                raise ValueError('Citation absent from evidence')
            source = by_id[sid]
            chunk = self.corpus.get_chunk(source['chunk_id'])
            citations.append({'id': chunk.id, 'source_id': sid, 'doc_id': chunk.doc_id, 'title': source['title'],
                              'clause': ('фрагмент' if source.get('display_label') else chunk.clause),
                              'source_clause': chunk.clause,
                              'display_label': source.get('display_label', source['title'] + ', ' + chunk.clause),
                              'quote': chunk.text,
                              'start': chunk.start, 'end': chunk.end})
        if generated.get('mode') == 'llm':
            # Keep internal evidence IDs and exact quotes for validation/UI;
            # only the final answer string receives readable source locators.
            citation_sources = [dict(source, citation_metadata=self.corpus.get_document(source['doc_id']).metadata)
                                for source in sources]
            generated['answer'] = render_author_answer(generated['answer'], citation_sources,
                                                        generated.get('used_ids', []))
        next_pending = None
        if generated.get('missing_facts'):
            # This records what was requested, not a fact or legal conclusion.
            from clarification import Clarification
            requested = tuple(str(item)[:512] for item in generated['missing_facts'][:8])
            next_pending = Clarification('Уточните: ' + '; '.join(requested),
                                         'model_clarification', requested)
        return {**generated, 'question': ctx['latest'], 'citations': citations,
                'dialogue_state': bind_pending(dialogue_state, next_pending),
                'warnings': list(plan.get('warnings', [])) + list(generated.get('warnings', [])),
                'trace': {'engine': 'v12_semantic', 'api_configured': self.client.enabled,
                          'normalization': normalization_trace,
                          'model_calls': normalization_trace.get('calls', 0) + generated.get('trace', {}).get('calls', 0),
                          'total_ms': round((time.monotonic() - started) * 1000),
                          'source_count': len(sources), 'retrieval': plan.get('trace', {}),
                          'generation': generated.get('trace', {})}}
