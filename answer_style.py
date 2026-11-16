"""Presentation of already-validated citations as ordinary document references.

This module never rewrites an answer's claims or selects evidence. It accepts
only the generator's exact, known-ID bibliography; anything unexpected leaves
the original answer untouched. Source IDs remain available in structured data.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import date
import re

_ID = re.compile(r'\bS\d{1,5}\b')
_CITATION = re.compile(r'\[\s*S\d{1,5}(?:\s*[,;]\s*S\d{1,5})*\s*\]')
_CITATION_SPACES = re.compile(r'[ \t]*' + _CITATION.pattern +
                              r'(?:[ \t]*' + _CITATION.pattern + r')*[ \t]*')
_CONTRACT = re.compile(r'\b([А-ЯЁA-Z]{1,12})[-‐‑–—](\d{1,12})[/\-‐‑–—](20\d{2})\b')


def _one_line(value: str) -> str:
    return re.sub(r'\s+', ' ', value).strip()


def _date(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            return date.fromisoformat(value).strftime('%d.%m.%Y')
        if re.fullmatch(r'\d{2}\.\d{2}\.\d{4}', value):
            day, month, year = map(int, value.split('.'))
            return date(year, month, day).strftime('%d.%m.%Y')
    except ValueError:
        pass
    return None


def _law_label(title: str) -> str | None:
    # A corpus chunk can inherit a numbered-list heading. Cite the article
    # supplied by the document identity, never infer a legal part/subparagraph
    # merely from that inherited number or a visible numeral in the body.
    article = re.search(r'\bст(?:атья|\.)?\s*(\d+(?:\.\d+)*)\b', title, re.I)
    if not article:
        return None
    code = re.search(r'\b(ТК|НК|ГК|УК|КоАП)(?:\s+РФ)?\b', title, re.I)
    if code:
        name = 'КоАП' if code[1].lower() == 'коап' else code[1].upper()
        return f'{name} РФ, ст. {article[1]}'
    federal = re.search(r'\b(\d+)\s*[-‐‑–—]?\s*ФЗ\b', title, re.I)
    if federal:
        return f'{federal[1]}-ФЗ, ст. {article[1]}'
    return None


def _document_label(source: dict, document_sources: list[dict]) -> tuple[str, bool]:
    title = _one_line(source['title'])
    law = _law_label(title)
    if law:
        return law, True
    text = '\n'.join(item['text'] for item in document_sources)
    metadata = source.get('citation_metadata')
    metadata = metadata if isinstance(metadata, dict) else {}

    if re.match(r'^регламент\b', title, re.I):
        revision = _date(metadata.get('revision_date'))
        if revision is None:
            dates = list(dict.fromkeys(match[1] for match in re.finditer(
                r'\bредакция\s+от\s+(\d{2}\.\d{2}\.\d{4})', text, re.I)
                if _date(match[1])))
            year = re.search(r'\bред(?:\.|акция)?\s+(20\d{2})\b', title, re.I)
            if year:
                dates = [value for value in dates if value.endswith(year[1])]
            if len(dates) == 1:
                revision = dates[0]
        if revision:
            # Preserve the document name and any explicit obsolete marker;
            # replace only its year-level edition suffix with the known date.
            title = re.sub(r'\s+ред(?:\.|акция)?\s+20\d{2}\b', '', title, flags=re.I)
            title = f'{title} (редакция от {revision})'
        return title, False

    if re.match(r'^служебная\s+записка\b', title, re.I):
        identifier = re.search(r'\bСЛУЖЕБНАЯ\s+ЗАПИСКА\s*№\s*([^\s,;]+)'
                               r'(?:\s+от\s+(\d{2}\.\d{2}\.\d{4}))?', text, re.I)
        if identifier:
            title = 'Служебная записка № ' + identifier[1]
            when = _date(identifier[2]) if identifier[2] else None
            if when:
                title += ' от ' + when
        return title, False

    contract = _CONTRACT.search(title)
    if contract:
        number = f'{contract[1]}-{contract[2]}/{contract[3]}'
        if re.match(r'^договор\b', title, re.I):
            prefix = title[:contract.start()].strip().rstrip('№').strip()
            title = prefix + ' № ' + number
        else:
            title = title[:contract.start()] + number + title[contract.end():]
    title = re.sub(r'^Доп\.?\s+соглашение\s+(\d+)\b',
                   r'Дополнительное соглашение № \1', title, flags=re.I)
    return title, False


def _footer(ids: list[str], safe: dict[str, dict]) -> str:
    return 'Источники: ' + '; '.join(
        '[' + key + '] ' + (safe[key].get('display_label') or
                            safe[key]['title'] + ', ' + safe[key]['clause'])
        for key in ids) + '.'


def _sort_clause(value: str):
    return tuple(int(part) for part in value.split('.'))


def render_author_answer(answer: str, sources: list[dict], used_ids: list[str]) -> str:
    """Return the same claims with references at the original citation positions.

    The function is deliberately fail-closed: invalid sources, unknown IDs,
    malformed markers, changed bibliographies or ID-order mismatches return
    ``answer`` unchanged. No-source clarifications/abstentions are unchanged.
    ``citation_metadata`` is optional canonical document metadata supplied by
    the caller; without it, exact edition dates require source-text evidence.
    """
    if not isinstance(answer, str) or not isinstance(used_ids, (list, tuple)) or not used_ids:
        return answer
    if not isinstance(sources, (list, tuple)) or any(not isinstance(key, str) for key in used_ids):
        return answer
    safe: dict[str, dict] = {}
    documents: dict[str, list[dict]] = {}
    for source in sources:
        if (not isinstance(source, dict) or source.get('suspicious')
                or not isinstance(source.get('id'), str)
                or not re.fullmatch(r'S\d{1,5}', source['id'])
                or source['id'] in safe
                or any(not isinstance(source.get(field), str) for field in ('title', 'clause', 'text'))
                or (source.get('display_label') is not None and
                    not isinstance(source['display_label'], str))):
            return answer
        safe[source['id']] = source
        identity = source.get('doc_id') or source['title']
        if not isinstance(identity, str):
            return answer
        if identity in documents and documents[identity][0]['title'] != source['title']:
            return answer
        documents.setdefault(identity, []).append(source)
    ids = list(dict.fromkeys(used_ids))
    if len(ids) != len(used_ids) or any(not isinstance(key, str) or key not in safe for key in ids):
        return answer
    suffix = '\n\n' + _footer(ids, safe)
    if not answer.endswith(suffix):
        return answer
    body = answer[:-len(suffix)]
    refs = list(dict.fromkeys(_ID.findall(body)))
    if refs != ids:
        return answer
    # No unknown/malformed machine references may disappear with formatting.
    for marker in re.findall(r'\[[^\[\]]*\]', body):
        if re.search(r'\bS\d', marker) and not _CITATION.fullmatch(marker):
            return answer
    # A valid-looking ID nested inside another bracket expression is not a
    # canonical citation and must not be silently deleted.
    depth = 0
    for position, char in enumerate(body):
        if char == '[':
            if depth and _CITATION.match(body, position):
                return answer
            depth += 1
        elif char == ']':
            depth = max(0, depth - 1)
    cleaned = _CITATION.sub('', body)
    if _ID.search(cleaned):
        return answer
    # Resolve labels once per document. Distinct documents must not become
    # indistinguishable merely because a normalized title drops a party name.
    cited_documents = list(dict.fromkeys(safe[key].get('doc_id') or safe[key]['title'] for key in ids))
    document_labels = {identity: _document_label(documents[identity][0], documents[identity])
                       for identity in cited_documents}
    collisions = {}
    for identity, (label, _) in document_labels.items():
        collisions.setdefault(label, []).append(identity)
    for identities in collisions.values():
        if len(identities) < 2:
            continue
        titles = [_one_line(documents[identity][0]['title']) for identity in identities]
        if len(set(titles)) != len(titles):
            return answer
        for identity, title in zip(identities, titles):
            document_labels[identity] = (title, False)

    def labels_for(group_ids, active_document):
        groups: OrderedDict[str, dict] = OrderedDict()
        for key in group_ids:
            source = safe[key]
            identity = source.get('doc_id') or source['title']
            label, law = document_labels[identity]
            group = groups.setdefault(identity, {'label': label, 'law': law, 'clauses': []})
            clause = source['clause'].strip()
            if not law and clause and clause.lower() not in {'текст', 'фрагмент'}:
                if clause not in group['clauses']:
                    group['clauses'].append(clause)
        labels = []
        sole_identity = next(iter(groups)) if len(groups) == 1 else None
        for identity, group in groups.items():
            label, clauses = group['label'], group['clauses']
            locator = ''
            if clauses:
                if all(re.fullmatch(r'\d+(?:\.\d+)*', clause) for clause in clauses):
                    locator = 'п. ' + ', '.join(sorted(clauses, key=_sort_clause))
                else:
                    locator = ', '.join(clauses)
            elif group['law']:
                article = re.search(r'ст\. \d+(?:\.\d+)*$', label)
                locator = article[0] if article else ''
            if locator and identity == sole_identity == active_document:
                labels.append(locator)
            else:
                labels.append(label + (', ' + locator if locator and not group['law'] else ''))
        return '; '.join(labels), sole_identity

    # Preserve the model's claim-to-source mapping. Only adjacent markers at
    # exactly the same citation position can be merged. Never aggregate
    # references across separate sentences, steps or paragraphs.
    pieces = []
    cursor = 0
    active_document = None
    for match in _CITATION_SPACES.finditer(body):
        between = body[cursor:match.start()]
        if '\n' in between:
            active_document = None
        group_ids = list(dict.fromkeys(_ID.findall(match[0])))
        label, active_document = labels_for(group_ids, active_document)
        suffix_punctuation = ''
        # Legacy drafts use 'Claim. [S001]'. Move only an ordinary terminal
        # period; decimal points, known abbreviations and ellipses stay intact.
        if (between.endswith('.') and not between.endswith('..')
                and not re.search(r'\b(?:ст|п|ч|абз|ред|г|руб|коп|тыс|млн|млрд|им|т|д|др|пр|рис|табл)\.$', between, re.I)
                and not re.search(r'\b[А-ЯЁA-Z]\.$', between)):
            between = between[:-1]
            suffix_punctuation = '.'
        right = body[match.end():]
        leading = '' if not between or between.endswith(('\n', ' ', '\t')) else ' '
        trailing = (' ' if right and not right.startswith(('\n', '.', ',', ';', ':', '!', '?', ')', '»'))
                    and match[0][-1] in ' \t' else '')
        pieces.append(between + leading + '(' + label + ')' + suffix_punctuation + trailing)
        cursor = match.end()
    pieces.append(body[cursor:])
    return ''.join(pieces)
