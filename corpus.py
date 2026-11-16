"""Local corpus loading and field-aware Russian lexical retrieval (BM25F).

Offsets are Unicode character offsets into ``documents[doc_id].text``;
the source text is never normalized or rewritten. Search normalization exists
only in the in-memory index. Retrieved document text is untrusted evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
import math
from pathlib import Path
import re
from typing import Iterable
from zipfile import ZipFile

from chunk import Chunk
from document import Document


_MONTHS = {
    "января": "01", "февраля": "02", "марта": "03", "апреля": "04",
    "мая": "05", "июня": "06", "июля": "07", "августа": "08",
    "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12",
}
_DATE = r"(?:\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}\s+(?:" + "|".join(_MONTHS) + r")\s+\d{4})"
_CONTRACT = re.compile(r"\b([А-ЯЁA-Z]{1,12})[-‐‑–—]\s*(\d{1,12})\s*[/\-‐‑–—]\s*(20\d{2})\b", re.I)
_NUMBERED = re.compile(r'^\s*[«"“]?\s*(\d+(?:\.\d+)*)([.)])\s+')
_STOP = set("и в во на по с со к ко от до за для не ни но а или о об это как что какой какая какие какого когда где ли можно нужно нужно бы был была были будет из у я мы вы он она они его ее их при чем чтобы также только уже ещё еще всем любой все всех через после перед между чем этот эта эти того той так же да нет мне нам вас ваш наш".split())
_STOP.update("договор договора договору договором договоре договоры договоров".split())
_ENDINGS = re.compile(r"(?:иями|ями|ами|ого|ему|ому|ыми|ими|его|ение|ения|ений|ению|ать|ять|ить|ий|ый|ой|ая|яя|ое|ее|ые|ие|ам|ям|ах|ях|ов|ев|ию|ия|ья|ью|ую|юю|ии|ом|ем|ым|им|ою|ею|а|я|ы|и|у|ю|е|о)$")
_CONCEPTS = (
    (re.compile(r"^(?:оплат|оплач|платеж|платит|платим|платить|расчет)"), "@payment"),
    (re.compile(r"^(?:достав|постав|привез|привоз|отгруз)"), "@delivery"),
    (re.compile(r"^(?:соглас(?:ов|у)|одобр)"), "@approval"),
    (re.compile(r"^согласи"), "@consent"),
    (re.compile(r"^(?:отозв|отзыв)"), "@revocation"),
    (re.compile(r"^(?:гарант)"), "@guarantee"),
    (re.compile(r"^(?:растор|прекращ|съезж|съех)"), "@termination"),
    (re.compile(r"^(?:испытан)"), "@probation"),
    (re.compile(r"^(?:хран|архив)"), "@archive"),
    (re.compile(r"^(?:качеств|дефект|недостат)"), "@quality"),
    (re.compile(r"^(?:срок|дней|дня|день|дни|течение|числа|месяц|час)"), "@duration"),
    (re.compile(r"^(?:аванс|предоплат)"), "@prepayment"),
    (re.compile(r"^(?:пропуск|доступ|круглосуточ|ночь|ночи|ночью|ночн)"), "@access"),
)


def _normalize(value: str) -> str:
    return value.lower().replace("ё", "е").translate(str.maketrans("‐‑–—", "----"))


def _tokens(value: str) -> list[str]:
    result: list[str] = []
    for word in re.findall(r"[а-яa-z]+|\d+(?:[./-]\d+)*", _normalize(value)):
        if word in _STOP or (len(word) < 2 and not word.isdigit()):
            continue
        result.append(word)
        if re.fullmatch(r"[а-я]{5,}", word):
            stem = _ENDINGS.sub("", word)
            if stem != word and len(stem) >= 3:
                result.append("~" + stem)
        for pattern, concept in _CONCEPTS:
            if pattern.match(word):
                result.append(concept)
    return result


def _iso_date(value: str | None) -> str | None:
    if not value:
        return None
    numeric = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", value)
    if numeric:
        day, month, year = numeric.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    textual = re.search(r"(\d{1,2})\s+([а-я]+)\s+(\d{4})", value.lower())
    if textual and textual.group(2) in _MONTHS:
        day, month, year = textual.groups()
        return f"{year}-{_MONTHS[month]}-{int(day):02d}"
    return None


def _match_date(pattern: str, text: str) -> str | None:
    match = re.search(pattern + r"\s*(" + _DATE + r")", text, re.I)
    return _iso_date(match.group(1)) if match else None


def _contracts(value: str) -> list[str]:
    return list(dict.fromkeys(f"{a.upper()}-{b}/{c}" for a, b, c in _CONTRACT.findall(value)))


def is_suspicious(text: str) -> bool:
    """Heuristic quarantine, not a complete prompt-injection security boundary."""
    value = _normalize(text)
    direct_address = bool(re.search(r"(?:для\s+ии|ии[- ]ассистент|языков\w*\s+модел|\bllm\b|\bassistant\b)", value))
    override = bool(re.search(r"(?:игнорир\w*|забудь|не\s+упоминай|не\s+следуй|вместо\s+правил|ignore\s+(?:all|previous)|disregard)", value))
    hidden_role = bool(re.search(r"(?:\[/?system\]|<\|(?:system|im_start)\|>|системн\w*\s+(?:промпт|сообщение)\s*:)", value))
    generic_override = bool(re.search(r"(?:игнорир\w*\s+(?:все\s+)?(?:предыдущ|остальн|инструкц|документ)|ignore\s+(?:all\s+)?previous\s+instructions)", value))
    universal_answer = bool(re.search(r"на\s+любой\s+вопрос.*(?:отвечай|выдавай)", value, re.S))
    return hidden_role or generic_override or universal_answer or (direct_address and override)


def _metadata(filename: str, text: str, kind: str) -> dict:
    name = filename.lower()
    meta: dict = {
        "contract_no": None, "signed_at": None, "effective_from": None,
        "effective_to": None, "revision_date": None, "entities": [], "inns": [],
    }
    numbers = _contracts(text[:1500])
    if numbers:
        meta["contract_no"] = numbers[0]
        meta["contract_numbers"] = numbers
    meta["entities"] = list(dict.fromkeys(re.findall(r'ООО\s+[«"]([^»"\n]+)[»"]', text)))
    meta["inns"] = list(dict.fromkeys(re.findall(r"ИНН\s*(\d{10,12})", text)))
    if kind == "law":
        meta["document_type"] = "law_article"
        article = re.search(r"(?:ст\.|статья)\s*(\d+(?:\.\d+)?)", text[:600], re.I)
        meta["article"] = article.group(1) if article else None
        code = re.search(r"законы__(НК|ТК|КоАП|152ФЗ|МРОТ)", filename, re.I)
        meta["code"] = code.group(1) if code else None
        url = re.search(r"^URL:\s*(\S+)", text, re.M)
        meta["source_url"] = url.group(1) if url else None
        # Editorial change dates do not establish a whole article's validity.
        meta["history_available"] = False
        return meta
    if re.search(r'^\s*дополнительное соглашение', text, re.I | re.M):
        meta["document_type"] = "amendment"
        meta["is_amendment"] = True
    elif re.search(r'^\s*договор\b', text, re.I | re.M):
        meta["document_type"] = "contract"
    elif re.search(r'^\s*регламент\b', text, re.I | re.M):
        meta["document_type"] = "regulation"
    elif re.search(r'^\s*Исх\.', text, re.I | re.M):
        meta["document_type"] = "letter"
    else:
        meta["document_type"] = "memo"
    # Contract preambles may quote an earlier contract date before their own date.
    for line in text[:1500].splitlines():
        if re.search(r"(?:г\.|область).*[—–]", line):
            date = re.search(_DATE, line)
            if date:
                meta["signed_at"] = _iso_date(date.group(0))
                break
    if meta["document_type"] in {"letter", "memo"}:
        meta["signed_at"] = _match_date(r"\bот", text[:600])
    meta["revision_date"] = _match_date(r"Редакция\s+от", text)
    meta["effective_from"] = _match_date(r"Действует\s+с", text)
    meta["effective_to"] = _match_date(r"УТРАТИЛА\s+СИЛУ\s+с", text)
    meta["effective_to_exclusive"] = bool(meta["effective_to"])
    if meta.get("is_amendment"):
        clause = re.search(r"Изложить\s+пункт\s+(\d+(?:\.\d+)*)", text, re.I)
        meta["amends_clause"] = clause.group(1) if clause else None
        if "вступает в силу с даты подписания" in text.lower():
            meta["effective_from"] = meta["signed_at"]
        condition = re.search(r"[^\n]*[Сс]пецификации[^\n]*после[^\n]*", text)
        if condition:
            meta["application_condition"] = condition.group(0).strip()
            meta["specification_date_after"] = _match_date(r"после", condition.group(0))
    if meta["document_type"] == "letter":
        meta["effective_from"] = _match_date(r"(?:^|\n)С", text)
    return meta


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _content_role(text: str, kind: str, clause: str) -> str:
    """Separate transport metadata from evidence without discarding its source."""
    if kind != "law":
        return "body"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and all(re.match(r"^(?:Источник|URL|Дата выгрузки):", line) for line in lines):
        return "metadata"
    if "редакционная пометка" in clause:
        return "editorial"
    # A short article title has no operative number or additional sentence.
    # Only demote it: unusually concise legal wording must remain searchable.
    if len(lines) == 1:
        title = re.fullmatch(r"Статья\s+\d+(?:\.\d+)?\.\s+([^\d.;:]{1,150})", lines[0])
        if title:
            return "heading"
    return "body"


def _split_long(text: str, start: int, end: int, limit: int = 1800) -> Iterable[tuple[int, int]]:
    """Split oversized paragraphs with overlap; keep exact source ranges."""
    while end - start > limit:
        target = start + limit
        candidates = [text.rfind(mark, start + limit // 2, target) for mark in ("\n", ". ", "; ", " ")]
        cut = max(candidates)
        if cut <= start:
            cut = target
        elif text[cut:cut + 2] in {". ", "; "}:
            cut += 1
        a, b = _trim(text, start, cut)
        if b > a:
            yield a, b
        next_start = max(start + 1, cut - 180)
        next_space = text.find(" ", next_start, cut)
        start = next_space + 1 if next_space >= 0 else next_start
    a, b = _trim(text, start, end)
    if b > a:
        yield a, b


def _chunk_ranges(document: Document) -> Iterable[tuple[int, int, str, str, bool]]:
    text, kind = document.text, document.kind
    article = document.metadata.get("article")
    prefix = f"ст. {article}" if article else "текст"
    clause, section, block_kind = prefix, "", "body"
    top_law_clause = ""
    start = 0
    position = 0
    suspicious = False
    for line in text.splitlines(keepends=True):
        clean = line.strip()
        line_start, line_end = position, position + len(line)
        position = line_end
        if not clean:
            continue
        suspect = is_suspicious(clean)
        numbered = _NUMBERED.match(clean)
        editorial = kind == "law" and bool(re.match(
            r"(?:Пункт|Подпункт|Часть|Статья)\s+\d.*(?:изменен|изменён|дополнен|утрати|вступа)|Федеральн\w*\s+закон", clean, re.I))
        heading = kind == "law" and bool(re.match(r"Статья\s+\d+[.]?\s+", clean))
        boundary = suspect or suspicious or numbered or editorial or heading or block_kind == "editorial"
        if boundary and line_start > start:
            a, b = _trim(text, start, line_start)
            if b > a:
                yield a, b, clause, section, suspicious
            start = line_start
        if suspect:
            clause, block_kind = "подозрительный абзац", "suspicious"
        elif suspicious:
            clause, block_kind = prefix, "body"
        elif editorial:
            clause, block_kind = prefix + ", редакционная пометка", "editorial"
        elif heading:
            clause, block_kind = prefix, "heading"
        elif numbered:
            number, punctuation = numbered.groups()
            block_kind = "body"
            if kind == "company":
                clause = number
                if "." not in number and punctuation == "." and len(clean) < 120:
                    section = clean
            elif punctuation == ")":
                clause = f"{prefix}, п. {top_law_clause}, подп. {number}" if top_law_clause else f"{prefix}, подп. {number}"
            else:
                top_law_clause = number
                clause = f"{prefix}, п. {number}"
        elif block_kind == "editorial":
            # A following unnumbered legal paragraph is distinct from the note.
            clause, block_kind = prefix, "body"
        suspicious = suspect
    a, b = _trim(text, start, len(text))
    if b > a:
        yield a, b, clause, section, suspicious


class Corpus:
    """Immutable-at-load TXT corpus with an in-memory BM25 index."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        payloads = self._read_payloads(self.path)
        manifest = []
        total_bytes = 0
        for filename, raw in payloads:
            text = raw.decode("utf-8")
            if not text.strip():
                continue
            digest = sha256(raw).hexdigest()
            doc_id = "doc_" + sha256(filename.encode("utf-8")).hexdigest()[:16]
            if doc_id in self.documents:
                raise ValueError(f"Duplicate document filename: {filename}")
            title = Path(filename).stem.split("__", 1)[-1].replace("_", " ")
            kind = "law" if filename.lower().startswith("законы__") or text.lstrip("\ufeff").startswith("Источник:") else "company"
            snapshot = _match_date(r"Дата\s+выгрузки:", text)
            document = Document(
                id=doc_id, title=title, filename=filename, text=text,
                kind=kind, snapshot_date=snapshot, metadata=_metadata(filename, text, kind),
                sha256=digest,
            )
            self.documents[doc_id] = document
            total_bytes += len(raw)
            manifest.append(filename + ":" + digest)
            for start, end, clause, section, suspect in _chunk_ranges(document):
                for a, b in _split_long(text, start, end):
                    chunk_id = "chunk_" + sha256(f"{doc_id}:{a}:{b}:{text[a:b]}".encode("utf-8")).hexdigest()[:20]
                    content_role = _content_role(text[a:b], kind, clause)
                    self.chunks[chunk_id] = Chunk(
                        id=chunk_id, doc_id=doc_id, title=title, clause=clause,
                        text=text[a:b], start=a, end=b, suspicious=suspect,
                        section=section, kind=kind, content_role=content_role,
                        searchable=not suspect and content_role != "metadata",
                    )
        if not self.documents:
            raise ValueError(f"No nonempty UTF-8 TXT documents found in {self.path}")
        self._build_index()
        company_count = sum(d.kind == "company" for d in self.documents.values())
        suspicious_count = sum(c.suspicious for c in self.chunks.values())
        self.stats = {
            "documents": len(self.documents), "chunks": len(self.chunks),
            "document_count": len(self.documents), "chunk_count": len(self.chunks),
            "company_count": company_count, "law_count": len(self.documents) - company_count,
            "bytes": total_bytes, "characters": sum(len(d.text) for d in self.documents.values()),
            "words": sum(len(d.text.split()) for d in self.documents.values()),
            "suspicious_chunks": suspicious_count, "suspicious_chunk_count": suspicious_count,
            "indexed_chunks": len(self._lengths),
            "metadata_chunks": sum(c.content_role == "metadata" for c in self.chunks.values()),
            "snapshot_dates": sorted({d.snapshot_date for d in self.documents.values() if d.snapshot_date}),
            "corpus_hash": sha256("\n".join(manifest).encode("utf-8")).hexdigest(),
        }

    @staticmethod
    def _read_payloads(path: Path) -> list[tuple[str, bytes]]:
        archive = path / "knowledge-base.zip" if path.is_dir() else path
        if archive.is_file() and archive.suffix.lower() == ".zip":
            with ZipFile(archive) as package:
                # Read in place, never extract archive paths to the filesystem.
                names = sorted(n for n in package.namelist() if n.lower().endswith(".txt") and not n.startswith("__MACOSX/"))
                return [(Path(name).name, package.read(name)) for name in names]
        if path.is_dir():
            files = sorted(path.rglob("*.txt"))
            return [(file.relative_to(path).as_posix(), file.read_bytes()) for file in files if file.is_file()]
        if path.is_file() and path.suffix.lower() == ".txt":
            return [(path.name, path.read_bytes())]
        raise FileNotFoundError(f"Expected a TXT directory or knowledge-base.zip: {path}")

    def _build_index(self) -> None:
        # Keep the source body separate from repeated document/section labels:
        # a title match should not outweigh an operative provision.
        self._postings: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
        self._lengths: dict[str, int] = {}
        self._field_lengths: dict[str, tuple[int, int]] = {}
        for chunk_id, chunk in self.chunks.items():
            if chunk.suspicious or not chunk.searchable:
                continue
            body = _tokens(chunk.text)
            context = _tokens(chunk.title + " " + chunk.section + " " + chunk.clause)
            self._lengths[chunk_id] = len(body) + len(context)
            self._field_lengths[chunk_id] = (len(body), len(context))
            body_counts, context_counts = Counter(body), Counter(context)
            for token in body_counts.keys() | context_counts.keys():
                self._postings[token][chunk_id] = (body_counts[token], context_counts[token])
        self._avg_length = sum(self._lengths.values()) / max(len(self._lengths), 1)
        self._avg_fields = tuple(sum(lengths[i] for lengths in self._field_lengths.values()) /
                                 max(len(self._field_lengths), 1) for i in (0, 1))

    def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def search(self, query: str, k: int = 8, doc_ids: Iterable[str] | None = None,
               chunk_ids: Iterable[str] | None = None) -> list[Chunk]:
        """Return ranked chunk copies, excluding quarantined instructions.

        ``doc_ids=[]`` intentionally searches nothing. A single string ID is
        accepted too. An empty/irrelevant query never returns arbitrary chunks.
        """
        if not isinstance(query, str) or not query.strip() or k <= 0:
            return []
        allowed = {doc_ids} if isinstance(doc_ids, str) else set(doc_ids) if doc_ids is not None else None
        allowed_chunks = {chunk_ids} if isinstance(chunk_ids, str) else set(chunk_ids) if chunk_ids is not None else None
        if allowed == set() or allowed_chunks == set():
            return []
        scores: dict[str, float] = defaultdict(float)
        total = len(self._lengths)
        # Identifiers select/boost documents, not their short title paragraphs.
        # Otherwise repeating an entity in a preamble can outrank the actual rule.
        lexical_query = _CONTRACT.sub(" ", _normalize(query))
        matched_entity_documents = set()
        for document in self.documents.values():
            for entity in document.metadata.get("entities", []):
                pattern = r"(?<!\w)" + re.escape(_normalize(entity)) + r"(?!\w)"
                if re.search(pattern, _normalize(query)):
                    matched_entity_documents.add(document.id)
                    lexical_query = re.sub(pattern, " ", lexical_query)
        query_tokens = set(_tokens(lexical_query))
        # A query containing only an identifier still returns its document.
        if not query_tokens:
            query_tokens = set(_tokens(query))
        for token in query_tokens:
            posting = self._postings.get(token, {})
            inverse = math.log(1 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for chunk_id, frequencies in posting.items():
                chunk = self.chunks[chunk_id]
                if allowed is not None and chunk.doc_id not in allowed:
                    continue
                if allowed_chunks is not None and chunk_id not in allowed_chunks:
                    continue
                body_length, context_length = self._field_lengths[chunk_id]
                frequency = (frequencies[0] / (0.25 + 0.75 * body_length / max(self._avg_fields[0], 1)) +
                             0.35 * frequencies[1] / (0.75 + 0.25 * context_length / max(self._avg_fields[1], 1)))
                # Years/amounts are resolved as conditions by the answer layer;
                # a coincidental year in a contract header must not dominate.
                weight = 1.8 if token.startswith("@") else 0.25 if token[0].isdigit() else 1.0
                scores[chunk_id] += weight * inverse * frequency * 2.2 / (frequency + 1.2)
        contract_numbers = set(_contracts(query))
        clause_match = re.search(r"(?:пункт\w*|п\.)\s*(\d+(?:\.\d+)*)", query, re.I)
        article_match = re.search(r"(?:стать\w*|ст\.)\s*(\d+(?:\.\d+)?)", query, re.I)
        normalized_query = _normalize(query)
        for chunk_id in list(scores):
            chunk = self.chunks[chunk_id]
            meta = self.documents[chunk.doc_id].metadata
            if contract_numbers:
                scores[chunk_id] *= 1.8 if meta.get("contract_no") in contract_numbers else 0.35
            if clause_match and (chunk.clause == clause_match.group(1) or f"п. {clause_match.group(1)}" in chunk.clause):
                scores[chunk_id] += 6.0
            if article_match and meta.get("article") == article_match.group(1):
                scores[chunk_id] += 8.0
            # Exact entity identity is more valuable than a shared name prefix.
            if chunk.doc_id in matched_entity_documents:
                scores[chunk_id] += 3.0
            if chunk.content_role == "heading":
                scores[chunk_id] *= 0.2
            elif chunk.content_role == "editorial" and not re.search(r"редакц|измен|дополн|вступ|утрат", normalized_query):
                scores[chunk_id] *= 0.5
        ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
        return [replace(self.chunks[cid], score=round(scores[cid], 6)) for cid in ordered if scores[cid] > 0]
