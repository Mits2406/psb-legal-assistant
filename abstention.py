"""Conservative recognition of source-scoped inability to answer.

This validates a speech act, not the truth of a model's retrieval assessment.
It never accepts legal conclusions merely because they follow a cautionary
prefix. Corpus-wide absence is narrowed to the provided sources by default.
"""
from __future__ import annotations

import re


_SCOPE = (r'(?:в\s+)?(?:(?:предоставленн\w*|представленн\w*|доступн\w*|'
          r'приложенн\w*|имеющ\w*|данн\w*)\s+)?'
          r'(?:источник\w*|материал\w*|документ\w*|фрагмент\w*|'
          r'баз[аеуы]\s+знаний|баз[аеуы])')
_EPISTEMIC = (r'(?:ответить|назвать|определить|установить|рассчитать|подтвердить|'
              r'проверить|сообщить|дать\s+(?:точный|достоверный|обоснованный)\s+ответ|'
              r'выбрать\s+формулу)')
_NOUN = (r'(?:документ\w*|источник\w*|текст\w*|копи\w*|экземпляр\w*|'
         r'акт\w*|талон\w*|справк\w*|сведени\w*|информаци\w*|данн\w*|'
         r'дат\w*|услови\w*|положени\w*|правил\w*|основани\w*|размер\w*|'
         r'сумм\w*|срок\w*|расчет\w*|договор\w*|приложени\w*|редакци\w*|'
         r'спецификаци\w*|подтверждени\w*|норм\w*)')
_OBJECT = r'(?:(?:[а-я]{3,}(?:ый|ий|ой|ая|яя|ое|ее|ые|ие|ого|его|ую|юю|ых|их)|эт[иоа]|этих)\s+){0,5}' + _NOUN
_FORBIDDEN = re.compile(
    r'\b(?:вправе|обязан\w*|должен|должна|должны|запрещ\w*|разреш\w*|'
    r'допуска\w*|положен\w*|освобожд\w*|составля\w*|равен|равна|равно|равны|'
    r'выплачива\w*|оплачива\w*|облага\w*|применя\w*|считается|признается|'
    r'устанавливается|требуется|предусмотрен\w*|необходимо|надо|'
    r'оплатите|перечислите|подпишите|согласуйте|удалите|увольте|'
    r'отменен\w*|выполнен\w*|согласован\w*|оплачен\w*|перечислен\w*)\b'
    r'|\bне\s+(?:нужно|нужна|нужен|нужны|надо)\b'
    r'|\b\d+(?:[.,]\d+)?\s*(?:%|процент\w*|рубл\w*|месяц\w*|лет|дн\w*)'
    r'|\b(?:штраф|неустойка|плата|ставка|срок|сумма)\s*[—–:=]\s*\S', re.I)
_PREFIX = re.compile(r'^\s*недостаточно\s+(?:данных|информации)\s*:\s*', re.I)


def _plain(value: str) -> str:
    return re.sub(r'\s+', ' ', value.lower().replace('ё', 'е')).strip(' \t\r\n.!?;')


def _safe_tail(value: str) -> bool:
    """A bounded nominal topic must not smuggle a second proposition."""
    return bool(value and len(value) <= 650 and not _FORBIDDEN.search(value)
                and not re.search(r'[<>\[\]{}=,:;—–]|https?://|\b(?:следовательно|значит|поэтому|однако|но|зато|хотя)\b', value)
                and not re.search(r'\b(?:я|мы|вы|он|она|они|оно|нет|что|котор\w*)\b', value)
                # Coordination must remain a nominal oblique phrase, not
                # a new subject such as '... и договор недействителен'.
                and not re.search(r'\b(?:и|а|или)\s+(?![а-я-]+(?:ам|ям|ах|ях|ов|ев|ом|ем|ии|ия|ию|ой|ей|ую|юю|ых|их|ым|им|ы|и|е)\b)', value)
                and not re.search(r'\b(?:не)?(?:действителен|действительна|действительно|действительны|'
                                  r'законен|законна|законно|законны|правомерен|правомерна|правомерно|правомерны|'
                                  r'минимален|минимальна|минимально|минимальны|допустим|допустима|допустимо|допустимы)\b', value)
                and not re.search(r'\b(?:можно|нельзя|нужно)\s+(?!' + _EPISTEMIC + r'\b)', value)
                and not re.search(r'\b[а-я]{3,}(?:ется|ются|ается|яются|ируют|ируется)\b', value))


def _source_limit(unit: str) -> bool:
    # Examples: 'В источниках нет сведений о размере X'; 'В базе знаний
    # отсутствуют документы по Y'. The arbitrary topic remains nominal.
    match = re.fullmatch(_SCOPE + r'\s+(?:нет|отсутству(?:ет|ют)|не\s+найден[оыа]?)\s+(.+)', unit)
    if match and re.match(_OBJECT, match[1]) and _safe_tail(match[1]):
        return True
    match = re.fullmatch(_SCOPE + r'\s+(?:не\s+(?:указан[оыа]?|приведен[оыа]?|описан[оыа]?|'
                         r'представлен[оыа]?|содержится)|отсутствует)\s+(.+)', unit)
    if match and re.match(_OBJECT, match[1]) and _safe_tail(match[1]):
        return True
    match = re.fullmatch(r'(.+?)\s+' + _SCOPE + r'\s+не\s+(?:указан[оыа]?|приведен[оыа]?|описан[оыа]?|представлен[оыа]?)', unit)
    return bool(match and re.match(_OBJECT, match[1]) and _safe_tail(match[1]))


def _inability(unit: str) -> bool:
    # Only an inability to answer/infer; 'нельзя платить' is never epistemic.
    prefix = r'(?:(?:поэтому|потому|так\s+что)\s+)?'
    basis = r'(?:по\s+(?:этим\s+)?' + _SCOPE + r'|по\s+ним|по\s+документам|без\s+этих\s+данных|без\s+них)'
    prefix += r'(?:' + basis + r'\s+)?'
    match = re.fullmatch(prefix + r'(?:я\s+)?(?:не\s+могу|невозможно|нельзя)\s+'
                         r'(?:(?:точно|достоверно|обоснованно)\s+)?' + _EPISTEMIC + r'(?:\s+(.+))?', unit)
    if match:
        return not match[1] or _safe_tail(match[1])
    # The entire clause must still end in inability. Entitlement wording is
    # permitted only as the unknown argument of an epistemic verb, never as
    # an assertion ('сумма вам положена') attached to a cautionary prefix.
    match = re.fullmatch(prefix + _EPISTEMIC + r'(.*?)\s+(?:я\s+)?(?:не\s+могу|невозможно|нельзя)', unit)
    if match:
        argument = re.sub(r'(?:,?\s+)' + basis + r'$', '', match[1]).strip(' ,')
        if not argument or _safe_tail(argument):
            return True
        indirect = re.fullmatch(r'(?:какая|какой|какое|какие)\s+(.+?)\s+(?:(?:вам|мне|нам|ему|ей|им)\s+)?положен[аоы]?', argument)
        if indirect:
            return bool(re.fullmatch(_OBJECT, indirect[1]) and _safe_tail(indirect[1]))
        nominal = re.fullmatch(r'положенн(?:ую|ый|ое|ые|ая)\s+(?:(?:вам|мне|нам|ему|ей|им)\s+)?(.+)', argument)
        if nominal:
            return bool(re.fullmatch(_OBJECT, nominal[1]) and _safe_tail(nominal[1]))
    match = re.fullmatch(prefix + _EPISTEMIC + r'(?:(?:\s+по\s+документам)|(?:\s+на\s+вопрос))?'
                         r'\s+(?:я\s+)?(?:не\s+могу|невозможно|нельзя)', unit)
    return bool(match)


def _document_need(unit: str) -> bool:
    unit = re.sub(r'^(?:для\s+(?:ответа|расчета|проверки|определения\s+\w+|точного\s+ответа)|'
                  r'чтобы\s+(?:ответить|уточнить|определить\s+\w+))\s*[,—–]?\s*', '', unit)
    match = re.fullmatch(r'(?:нужен|нужна|нужно|нужны|необходим[аоы]?|пришлите|предоставьте|покажите)\s+(.+)', unit)
    if not match or not re.match(_OBJECT, match[1]):
        return False
    parts = [part.strip() for part in match[1].split(',')]
    if not all(_safe_tail(part) for part in parts):
        return False
    # Additional list items must begin in an oblique nominal form. Do not
    # turn 'нужны данные, договор действителен' into a document request.
    return all(re.match(r'[а-я-]+(?:ого|его|ому|ему|ым|им|ых|их|ыми|ими|ах|ях|ам|ям|ов|ев|ии|ию|ом|ем)\b', part)
               and not re.search(r'\b[а-я]{3,}(?:ен|ена|ено|ены|ан|ана|ано|аны|л|ла|ли|ло)\b', part)
               for part in parts[1:])


def _supporting_sentence(unit: str) -> bool:
    if re.fullmatch(r'по\s+памяти\s+(?:я\s+)?отвечать\s+не\s+буду'
                    r'(?:,\s*(?:так\s+как|поскольку)\s+нормы\s+могли\s+измениться)?', unit):
        return True
    if _document_need(unit):
        return True
    # Referral is ancillary, never authorization for a business action.
    return bool(re.fullmatch(r'(?:для\s+(?:ответа|уточнения|проверки)\s+)?обратитесь\s+'
                             r'(?:к\s+юристу|к\s+специалисту|в\s+отдел\s+кадров|в\s+юридический\s+отдел)'
                             r'(?:\s+или\s+(?:к\s+юристу|к\s+специалисту|в\s+отдел\s+кадров))?', unit))


def inspect_abstention(text: str) -> bool:
    """True only for a whole uncited source-limitation speech act.

    This is intentionally narrower than a natural-language legal verifier.
    Requests without a limitation remain the existing clarification parser's
    responsibility. A single rejected sentence rejects the whole answer.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 2400:
        return False
    if re.search(r'[\[\]{}<>]|https?://|\bS\d{1,5}\b', text):
        return False
    explicit = bool(_PREFIX.match(text))
    value = _PREFIX.sub('', text.strip())
    # Preserve source references such as 'ст. 6' within a nominal request.
    value = re.sub(r'\b(ст|п|ч|ред|г)\.(?=\s*\d)', lambda m: m[1] + '\u2060', value, flags=re.I)
    units = [_plain(unit.replace('\u2060', '.')) for unit in re.split(r'(?<=[.!?])\s+|\n+|;\s*', value)]
    units = [unit for unit in units if unit]
    found_limit = explicit
    for unit in units:
        # An explanation may be attached by comma, but only another allowed
        # epistemic clause can follow the causal connector.
        parts = re.split(r',\s*(?=(?:поэтому|так\s+что)\b)', unit)
        for part in parts:
            limited = _source_limit(part) or _inability(part)
            if not limited and not _supporting_sentence(part):
                return False
            found_limit = found_limit or limited
    return bool(units and found_limit)


def normalize_abstention(text: str, *, complete_corpus: bool = False) -> str | None:
    """Return an accepted limitation with honest source scope, otherwise None.

    ``complete_corpus`` must mean the model actually inspected the whole
    applicable corpus, not merely that retrieval selected some full documents.
    """
    if not inspect_abstention(text):
        return None
    result = text.strip()
    if not complete_corpus:
        result = re.sub(r'\bв\s+(?:(?:предоставленной|представленной|доступной|данной|имеющейся)\s+)?'
                        r'базе(?:\s+знаний)?\b',
                        lambda match: 'В предоставленных источниках' if match[0][0].isupper() else 'в предоставленных источниках',
                        result, flags=re.I)
    return result
