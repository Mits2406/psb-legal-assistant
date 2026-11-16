"""A conservative, metadata-only clarification gate before corpus retrieval.

The gate never calls search or a model. Its catalog contains identities and
source-derived applicability/topic signatures prepared while loading the corpus.
It does not decide a legal answer or filter editions out of the later retrieval.
Conversation state is supplied by the caller; nothing is retained between rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from query import inspect_query, infer_topic, norm, temporal_mentions, MONTHS, _resolve_identity
from registry import contract_number


# Intent vocabulary, not corpus-specific facts. Only signatures of actual safe
# clauses are compared, so an unchanged rule does not trigger a date question.
_RULE_TOPICS = {
    'approval': r'соглас|одобр|инициатор|подписан|типов',
    'approval_initiator': r'инициатор',
    'approval_finance': r'финансов.{0,20}директор|поручител|залогодател|аффилирован',
    'approval_general': r'генеральн.{0,20}директор',
    'originals_transfer': r'оригинал.{0,60}переда|переда.{0,60}оригинал',
    'foreign_counterparty': r'иностранн|санкци|валютн|комплаенс',
    'archive': r'архив|хранени|выдач.{0,30}оригинал',
    'delivery_address': r'адрес|место.{0,20}постав|постав.{0,20}мест',
    'payment': r'оплат|платеж|платёж|расчет|расчёт',
    'rent': r'арендн.{0,20}плат|плат.{0,20}аренд',
    'warranty': r'гарант',
    'quality_acceptance': r'прием|приём|качеств',
    'defect_claim': r'претензи|дефект|недостат',
    'price_inclusions': r'цен|стоим|упаков|включен|включён',
    'contract_term': r'срок.{0,20}действ|продлен|продлён|прекращ',
    'access': r'доступ|пропуск|круглосуточ|режим',
    'return_premises': r'возврат|возвращ|освобод|акт.{0,20}помещ',
}
_CONTRACT_TOPICS = set(_RULE_TOPICS) - {
    'approval', 'approval_initiator', 'approval_finance', 'approval_general',
    'originals_transfer', 'foreign_counterparty', 'archive',
}
_BROAD_LEGAL = re.compile(
    r'\b(?:закон|законодательств|кодекс|гк\b|тк\b|нк\b|работник|работодател|'
    r'зарплат|заработн|налог|алименты|потребител)', re.I)
_COMPARISON = re.compile(
    r'сравн|сопостав|отлич(?:а|и)|различ|по обоим|по двум|в обоих|во всех|вс[её]х договор|'
    r'обе редакци|обоих редакци|вс[её] редакци|все редакци|до и после|'
    r'как.{0,25}(?:меня|измен)|стар.{0,30}нов|нов.{0,30}стар|'
    r'если.{0,80}(?:иначе|а если)|условн.{0,15}вариант|зависит.{0,25}дат', re.I)
_UNKNOWN_FACT = re.compile(
    r'не\s+(?:знаю|помню|известн)|неизвестн|нет\s+данных|'
    r'не\s+(?:располага\w*|владе\w*|облада\w*)|'
    r'не\s+могу\s+(?:сообщить|сказать|назвать|уточнить)|'
    r'(?:нет|отсутств\w*).{0,30}(?:информаци|сведени|данны|доступ)|'
    r'(?:затрудня\w*|сложно|трудно).{0,20}(?:сказать|ответить|сообщить)|'
    r'(?:сумм|дат|контрагент|предмет).{0,30}не\s*(?:определ|извест|уточнен|назван)|'
    r'(?:информаци\w*|сведени\w*|данны\w*)[^,.;!?\n]{0,180}\b(?:нет|отсутств\w*)|'
    r'ответ(?:ь|ьте)\s+условно|условн.{0,15}ветв', re.I)
_CURRENT_PERIOD = re.compile(r'\b(?:сейчас|сегодня|действующ\w*|текущ\w*)\b|на данный момент', re.I)


def _intent(text):
    topic = infer_topic(text)
    if topic == 'generic' and re.search(r'согласовать|согласовывать|согласу[ею]', text, re.I):
        return 'approval'
    return topic


def _scope(topic):
    if topic in _CONTRACT_TOPICS:
        return 'contract'
    if topic in _RULE_TOPICS:
        return 'internal_process'
    if topic.startswith('law'):
        return 'law'
    return None


def _elliptical(text):
    return bool(len(text) < 180 and re.search(
        r'^\s*(?:нет\b|(?:поправ|исправ|уточн|ошиб)[а-яё]*\b|а\s+если\b)'
        r'|\b(?:по нему|по ней|в этом случае|тот же|та же|те же|том же|тех же|меняется ответ)\b'
        r'|^\s*(?:а\s+)?(?:сколько\s+(?:дней|месяцев|лет|рублей)|в какой срок)\s*[?.!]*$', text, re.I))


def _topic(ctx):
    text = str(ctx.get('resolved_question') or ctx['text'])
    topic = ctx['topic']
    # The object of a request can be a regulation's approval procedure even
    # when a generic verb (for example, "продление") resembles a contract term.
    # Do not ask the user to identify a contract that the question never selects.
    if (topic in _CONTRACT_TOPICS and not ctx.get('contracts')
            and not ctx.get('ambiguous_entities') and not ctx.get('identity_conflict')
            and re.search(r'регламент', text, re.I)
            and re.search(r'согласу|согласов|одобр', text, re.I)):
        topic = 'approval'
    if topic.startswith('approval'):
        if re.search(r'финансов.{0,20}директор|финдиректор', text, re.I):
            return 'approval_finance'
        if re.search(r'генеральн.{0,20}директор', text, re.I):
            return 'approval_general'
        if re.search(r'инициатор|иницииру|(?:кто|кем).{0,25}(?:перв|начина|начать|запуск)', text, re.I):
            return 'approval_initiator'
        # Keep a specific intent already established by the router. Widening
        # it here makes unrelated differences between editions look material.
        return topic
    return topic


def _fact_reply(text, registry):
    """Identify short answers, not fresh questions, using user-authored facts."""
    value = text.strip()
    if len(value) > 260 or '?' in value:
        return False
    if re.search(r'\b(?:кто|что|сколько|когда|куда|почему|какой|какая|какие|каков|можно|нужно|расскажи|покажи|объясни)\b', value, re.I):
        return False
    if _UNKNOWN_FACT.search(value):
        return True
    if _CURRENT_PERIOD.search(value) and len(value) < 90 and _intent(value) == 'generic':
        return True
    identity = _resolve_identity(value, registry)
    if identity['found'] and _intent(value) == 'generic':
        return True
    if temporal_mentions(value):
        # A dated, independently stated substantive task is not a fact reply.
        return _intent(value) == 'generic' and not re.search(r'сравн|сопостав|проверить|перечисл|опиши', value, re.I)
    if re.fullmatch(r'(?:в\s+)?20\d\d(?:\s*г(?:од(?:у|а)?|\.)?)?[.! ]*', value, re.I):
        return True
    month_pattern = '|'.join(sorted(MONTHS, key=len, reverse=True))
    if re.fullmatch(r'(?:в\s+|на\s+|от\s+)?(?:\d{1,2}\s+)?(?:' + month_pattern + r')[а-я]*[.! ]*', value, re.I):
        return True
    return False


def conversation_context(question, history, event_date, registry, clarifier=None):
    """Resume the last substantive user task after a date/name/ID-only reply."""
    ctx = inspect_query(question, history, event_date, registry=registry, client=None)
    ctx['topic'] = _intent(ctx['text'])
    if len(ctx['turns']) < 2:
        return ctx
    reply = _fact_reply(ctx['latest'], registry)
    dependent = _elliptical(ctx['latest'])
    previous_topic = next((_intent(turn) for turn in reversed(ctx['turns'][:-1])
                           if not _fact_reply(turn, registry)), 'generic')
    # A fresh internal-process question after a rental/payment question does
    # not inherit that contract's event date. Explicit ellipsis keeps context.
    if (not reply and not dependent and _scope(ctx['topic']) is not None
            and _scope(previous_topic) is not None and _scope(ctx['topic']) != _scope(previous_topic)):
        fresh = inspect_query(ctx['latest'], None, event_date, registry=registry, client=None)
        fresh['topic'] = _intent(fresh['text'])
        fresh['history_scope'] = 'current_turn'
        return fresh
    if re.search(r'(?:друг|нов)[а-я]*\s+(?:парт|спецификац)', ctx['latest'], re.I):
        latest_only = inspect_query(ctx['latest'], None, event_date, registry=registry, client=None)
        ctx['spec_span'], ctx['spec_date'] = latest_only['spec_span'], latest_only['spec_date']
        ctx['clarification_reply'] = {'reset_field': 'specification_date', 'reason': 'different_batch'}
    if not reply and not (dependent and ctx['topic'] == 'generic'):
        return ctx
    task, task_index = None, None
    for index in range(len(ctx['turns']) - 2, -1, -1):
        turn = ctx['turns'][index]
        if not _fact_reply(turn, registry):
            task, task_index = turn, index
            break
    if task is None:
        return ctx
    replies = ctx['turns'][task_index + 1:]
    # Labels such as 'уточнение пользователя' are not query terms: they can
    # accidentally retrieve legal norms about clarification of personal data.
    # Keep raw chronological turns in context, and ground the actual task with
    # supplied facts for both retrieval and extractive ranking.
    fact_text = ' '.join(reply.strip(' .!?') for reply in replies)
    combined = task.rstrip(' .!?') + ' ' + fact_text + ('?' if task.rstrip().endswith('?') else '')
    ctx['_clarification_task'] = task
    ctx['_clarification_facts'] = tuple(replies)
    ctx['resolved_question'] = combined
    ctx['text'] = norm(combined)
    ctx['topic'] = _intent(ctx['text'])
    ctx['query'] = ctx['text'] + (' ' + ' '.join(ctx['contracts']) if ctx['contracts'] else '')
    # Reconstruct the preceding gate from user turns rather than trusting an
    # assistant's factual assertions. A bare date immediately after a request
    # for the specification date fills that slot, not a delivery-date slot.
    dates = temporal_mentions(ctx['latest'])
    bare_year = re.fullmatch(r'(?:в\s+)?(20\d\d)(?:\s*г(?:од(?:у|а)?|\.)?)?[.! ]*', ctx['latest'], re.I)
    if not dates and bare_year and len(replies) == 1:
        # A year is an interval, never an invented day. A later year after an
        # incomplete day/month is handled by the more precise branch below.
        from datetime import date
        year = int(bare_year[1])
        ctx['event_span'] = (date(year, 1, 1), date(year, 12, 31))
        ctx['event'] = None
        dates = [{'from': ctx['event_span'][0], 'to': ctx['event_span'][1]}]
    if (not dates and len(replies) > 1
            and re.fullmatch(r'(?:в\s+)?(20\d\d)(?:\s*г(?:од(?:у|а)?|\.)?)?[.! ]*', ctx['latest'], re.I)):
        year = re.search(r'20\d\d', ctx['latest'])[0]
        # A year supplied immediately after an incomplete month/day completes
        # that user's date, rather than borrowing a year from a contract ID.
        dates = temporal_mentions(replies[-2].strip(' .!?') + ' ' + year)
        if dates:
            ctx['event_span'] = (min(d['from'] for d in dates), max(d['to'] for d in dates))
            ctx['event'] = ctx['event_span'][0] if ctx['event_span'][0] == ctx['event_span'][1] else None
    if (clarifier is not None and dates and ctx.get('spec_span') is None
            and not re.search(r'поставк|доставк|отгрузк|оплат|упд|договор', ctx['latest'], re.I)):
        previous_turns = ctx['turns'][:-1]
        previous = conversation_context(previous_turns[-1],
            [{'role': 'user', 'content': turn} for turn in previous_turns[:-1]],
            None, registry, clarifier)
        requested = clarifier.check(previous)
        if requested is not None and requested.reason == 'specification_period':
            ctx['spec_span'] = (min(d['from'] for d in dates), max(d['to'] for d in dates))
            if ctx['spec_span'][0] == ctx['spec_span'][1]:
                ctx['spec_date'] = ctx['spec_span'][0]
            ctx['clarification_reply'] = {'field': 'specification_date', 'user_turn': len(ctx['turns']) - 1}
    return ctx


@dataclass(frozen=True)
class Clarification:
    question: str
    reason: str
    fields: tuple[str, ...]
    candidates: tuple[str, ...] = ()

    def response(self):
        return {
            'status': 'clarify', 'mode': 'clarification', 'answer': self.question,
            'used_ids': [], 'citations': [], 'warnings': [],
            'missing_facts': list(self.fields),
            'clarification': {'stage': 'pre_retrieval', 'reason': self.reason,
                              'fields': list(self.fields), 'candidates': list(self.candidates)},
            'self_check': {'enabled': True, 'status': 'not_applicable', 'completed': False, 'changed': False},
        }


class PreRetrievalClarifier:
    def __init__(self, registry):
        self.registry = registry
        self.contracts = [p for p in registry.profiles.values() if p.document_type == 'contract']
        self.families = {}
        for profile in registry.profiles.values():
            if profile.document_type == 'regulation' and profile.family:
                self.families.setdefault(profile.family, []).append(profile)
        self.signatures = {}
        self.clause_topics = {}
        self.clause_texts = {}
        self.letter_identifiers = {}
        for profile in registry.profiles.values():
            if profile.document_type == 'letter':
                identifiers = {contract_number(match) for proof in profile.evidence
                               if re.match(r'\s*Исх\.', proof.quote, re.I)
                               for match in re.findall(r'№\s*([А-ЯA-Z0-9]+(?:[-‐‑–—/][А-ЯA-Z0-9]+)*)', proof.quote, re.I)}
                if identifiers:
                    self.letter_identifiers[profile.doc_id] = identifiers
        for chunk in registry.corpus.chunks.values():
            if (not chunk.searchable or chunk.suspicious
                    or chunk.content_role in {'metadata', 'editorial', 'heading'}):
                continue
            text = re.sub(r'^\s*\d+(?:\.\d+)*[.)]?\s*', '', norm(chunk.text))
            self.clause_texts.setdefault((chunk.doc_id, str(chunk.clause)), []).append(text)
            for topic, pattern in _RULE_TOPICS.items():
                if re.search(pattern, text, re.I):
                    self.signatures.setdefault((chunk.doc_id, topic), set()).add(text)
                    self.clause_topics.setdefault((chunk.doc_id, str(chunk.clause)), set()).add(topic)

    @staticmethod
    def _numbers(profiles):
        return sorted({number for p in profiles for number in p.contract_numbers})

    def _contract_candidates(self, ctx):
        profiles = self.contracts
        text = ctx.get('resolved_question') or ctx['text']
        # Contract type is stated in the source header, not guessed from a file.
        kinds = [stem for stem in ('аренд', 'постав') if re.search(stem, text, re.I)]
        if kinds:
            narrowed = [p for p in profiles if any(
                any(re.search(stem, e.quote, re.I) for stem in kinds)
                for e in p.evidence if re.match(r'ДОГОВОР\b', e.quote, re.I))]
            if narrowed:
                profiles = narrowed
        return self._numbers(profiles)

    def _identity(self, ctx, topic):
        if topic not in _CONTRACT_TOPICS and not (
                topic == 'generic' and re.search(r'по\s+(?:(?:этому|нашему|данному)\s+)?договор|контрагент', ctx['text'], re.I)):
            return None
        text = ctx.get('resolved_question') or ctx['text']
        if _BROAD_LEGAL.search(text):
            return None
        known = {contract_number(n) for p in self.contracts for n in p.contract_numbers}
        if any(contract_number(n) not in known for n in ctx.get('explicit_contracts', ())):
            # A party may have other, absent contracts. Its known agreement
            # does not prove that an explicitly requested unknown ID is wrong.
            return None
        if ctx.get('identity_conflict'):
            candidates = sorted(set(ctx.get('contracts', ())) | {
                n for item in ctx.get('ambiguous_entities', ()) for n in item.get('candidates', ())})
            return Clarification('Уточните: укажите номер договора и полное наименование его контрагента — какие из названных вами данных верны?',
                                 'identity_conflict', ('Номер договора и его контрагент',), tuple(candidates))
        if ctx.get('contracts'):
            return None
        ambiguous = ctx.get('ambiguous_entities', [])
        candidates = sorted({n for item in ambiguous for n in item.get('candidates', ())})
        if not candidates:
            candidates = self._contract_candidates(ctx)
        if ambiguous or len(candidates) > 1:
            listing = ', '.join(candidates[:5])
            suffix = f' Варианты из каталога: {listing}.' if listing else ''
            return Clarification('Уточните: о каком договоре идёт речь? Укажите его номер или полное наименование контрагента.' + suffix,
                                 'contract_identity', ('Номер договора или полное наименование контрагента',), tuple(candidates))
        return None

    def _period(self, ctx, topic):
        text = ctx.get('resolved_question') or ctx['text']
        if (topic not in _RULE_TOPICS or ctx.get('event_span')
                or _CURRENT_PERIOD.search(text)
                or re.search(r'редакци\w*\s+(?:от\s+)?20\d\d\b|регламент\w*\s+(?:от\s+)?20\d\d\s*г', text, re.I)):
            return None
        if topic not in {'approval', 'approval_initiator', 'approval_finance', 'approval_general',
                         'originals_transfer', 'foreign_counterparty', 'archive'}:
            return None
        # A source identifier proven by the letter's header selects that
        # document regardless of the preposition used to refer to it. A
        # difference between regulation editions does not make its contents
        # ambiguous. Leave the letter's own applicability to the source text.
        normalized = text.upper().translate(str.maketrans('‐‑–—', '----'))
        named_letters = {doc_id for doc_id, identifiers in self.letter_identifiers.items()
                         if any(re.search(r'(?<![\w/-])' + re.escape(identifier) + r'(?![\w/-])', normalized)
                                for identifier in identifiers)} if re.search(r'письм', text, re.I) else set()
        letter_contents = len(named_letters) == 1 and topic not in {'approval_finance', 'approval_general'}
        if letter_contents or re.search(r'по\s+письм|в\s+письм|согласно\s+письм', text, re.I):
            return None
        named = re.findall(r'(?:ООО|АО|ПАО|ОАО|ЗАО|ИП)\s*[«"“]([^»"”\n]+)', text, re.I)
        families = []
        for family, profiles in self.families.items():
            if named and not any(norm(name) in family for name in named):
                continue
            signatures = [self.signatures.get((p.doc_id, topic), set()) for p in profiles]
            dated = [p for p in profiles if p.conditions]
            if len(dated) > 1 and any(signatures) and len({frozenset(s) for s in signatures}) > 1:
                families.append(family)
        if not families:
            return None
        return Clarification('Уточните: на какую дату или период нужно проверить условия регламента? Укажите также год.',
                             'edition_period', ('Дата или период события с указанием года',), tuple(families))

    def _amendment_affects_question(self, amendment, targets, text):
        # A delivery verb alone does not make an address amendment relevant
        # to a question about delivery/payment deadlines. Check both actual
        # versions of the changed clause before suppressing the date question:
        # an amendment that also changes timing must still be disambiguated.
        asks_time = re.search(r'срок|когда|сколько.{0,25}(?:дн|час|месяц)', text, re.I)
        asks_location = re.search(r'адрес|куда|где|мест[оу]\b|пункт.{0,20}(?:постав|достав)', text, re.I)
        if not asks_time or asks_location:
            return True
        originals = [' '.join(self.clause_texts.get((p.doc_id, amendment.target_clause), ())) for p in targets]
        replacement = ' '.join(self.clause_texts.get((amendment.doc_id, amendment.target_clause), ()))
        versions = [*originals, replacement]
        location_rule = r'адрес|по месту|место.{0,20}(?:постав|достав)'
        timing_rule = r'в течение|не позднее|не ранее|\bсрок|\bдн(?:ей|я|ь)\b|\bчас(?:ов|а|ы)?\b|\bмесяц'
        return not (versions and all(value and re.search(location_rule, value, re.I) for value in versions)
                    and not any(re.search(timing_rule, value, re.I) for value in versions))

    def _amendment_period(self, ctx, topic):
        if ctx.get('spec_span') or topic not in _CONTRACT_TOPICS:
            return None
        text = ctx.get('resolved_question') or ctx['text']
        if (re.search(r'что.{0,30}(?:написан|указан|содержит)|текст\s+пункт|формулировк|процитируй|приведи\s+пункт|покажи\s+пункт', text, re.I)
                and re.search(r'пункт|п\.\s*\d|основн.{0,15}договор', text, re.I)):
            return None
        requested = {contract_number(n) for n in ctx.get('contracts', ())}
        if not requested:
            return None
        for amendment in self.registry.amendments:
            if contract_number(amendment.contract_no) not in requested:
                continue
            if not any(c.field == 'specification_date' for c in amendment.conditions):
                continue
            targets = [p for p in self.contracts if amendment.contract_no in p.contract_numbers]
            relevant = any(topic in self.clause_topics.get((p.doc_id, amendment.target_clause), ()) for p in targets)
            if relevant and self._amendment_affects_question(amendment, targets, text):
                return Clarification('Уточните: какова дата подписания спецификации к этой партии товара? Нужна именно дата спецификации, включая год.',
                                     'specification_period', ('Дата подписания спецификации',), (amendment.contract_no,))
        return None

    def check(self, ctx):
        text = ctx.get('resolved_question') or ctx['text']
        semantic = ctx.get('semantic')
        if isinstance(semantic, dict) and semantic.get('validated') is True:
            # The normalizer supplies a typed task, not another sentence for
            # this regex router. Applicability still comes from the registry.
            topic = semantic.get('topic', _topic(ctx))
            if semantic.get('task_type') == 'general':
                return None
            if semantic.get('task_type') == 'comparison' and not ctx.get('identity_conflict'):
                return None
            identity = self._identity(ctx, topic)
            if identity is not None:
                unknown = {item.get('field') for item in semantic.get('unknown_facts', [])}
                if not unknown.intersection({'contract', 'counterparty', 'identity'}):
                    return identity
                # Identity has alternative identifying facts. Not remembering
                # the number does not mean the user cannot name the party (or
                # vice versa). Ask only for the still-available identifier.
                if 'identity' not in unknown:
                    if 'contract' in unknown and 'counterparty' not in unknown:
                        return Clarification('Уточните: укажите полное наименование контрагента по этому договору.',
                                             'contract_identity', ('Полное наименование контрагента',), identity.candidates)
                    if 'counterparty' in unknown and 'contract' not in unknown:
                        return Clarification('Уточните: укажите номер этого договора.',
                                             'contract_identity', ('Номер договора',), identity.candidates)
            if semantic.get('task_type') in {'document_text', 'comparison'}:
                return None
            unknown = {item.get('field') for item in semantic.get('unknown_facts', [])}
            amendment = None if 'specification_date' in unknown else self._amendment_period(ctx, topic)
            period = None if unknown.intersection({'event_date', 'period'}) else self._period(ctx, topic)
            return amendment or period
        # Explicit comparison/multi-scenario requests should receive all their
        # branches; clarification must not erase a fully answerable comparison.
        if _COMPARISON.search(text) or (
                re.search(r'редакци', text, re.I)
                and len(set(re.findall(r'\b20\d\d\b', text))) > 1):
            return None
        topic = _topic(ctx)
        # Repeating a question after an explicit 'I do not know' cannot resolve
        # it. Let generation explain source-supported conditional branches or
        # name the missing document, without selecting a branch by guesswork.
        if _UNKNOWN_FACT.search(text):
            return None
        return self._identity(ctx, topic) or self._amendment_period(ctx, topic) or self._period(ctx, topic)
