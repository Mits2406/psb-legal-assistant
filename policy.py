"""Interpret evidence-backed document rules, independent of corpus identifiers."""
from __future__ import annotations

from dataclasses import asdict
from datetime import date
import re

from query import (contracts_in, dates_in, inspect_query, norm, resolve_context,
                   temporal_mentions, user_turns)
from registry import RuleRegistry


def _result(chunks=(), intro='', status='answer', missing=(), rules=(), forced=True):
    return {'chunks':list({c.id:c for c in chunks}.values()), 'intro':intro,
            'status':status, 'missing':list(missing), 'rules':list(rules), 'forced':forced}


def evaluate_condition(condition, ctx):
    """True/False/None: a whole date interval must satisfy a supported predicate."""
    span=ctx.get('spec_span') if condition.field=='specification_date' else ctx.get('event_span') if condition.field=='event_date' else None
    if span is None:return None
    try:boundary=date.fromisoformat(condition.value)
    except (ValueError,TypeError):return None
    checks={'gt':lambda d:d>boundary, 'gte':lambda d:d>=boundary,
            'lt':lambda d:d<boundary, 'lte':lambda d:d<=boundary}
    check=checks.get(condition.op)
    if check is None:return None
    first,last=check(span[0]),check(span[1])
    return first if first==last else None


def evaluate_conditions(conditions,ctx):
    values=[evaluate_condition(c,ctx) for c in conditions]
    if False in values:return False
    return None if None in values else True


def _trace_rule(rule, result, selection_reason=None):
    return {'doc_id':rule.doc_id,'decision':result,
            'selection_reason':selection_reason,
            'conditions':[asdict(c) for c in rule.conditions],
            'evidence':[asdict(e) for e in rule.evidence]}


def _chunks(corpus,doc_id):
    return [c for c in corpus.chunks.values() if c.doc_id==doc_id and not c.suspicious and c.searchable]


def _topic_chunks(corpus,doc_id,topic,question=''):
    """Vocabulary describes concepts; no document-specific clause numbers."""
    def matches(chunk):
        text=norm(chunk.text)
        if topic=='payment':
            return (bool(re.search(r'оплачива|оплат[а-я]*\s+(?:товар|производ|осуществ|внос)|срок\s+оплат|предоплат',text))
                    and not re.search(r'неустойк|пени|просрочк',text))
        if topic=='delivery_address':return bool(re.search(r'достав|постав|отгруз',text) and re.search(r'адрес|куда',text))
        if topic=='warranty':return bool(re.search(r'гарантийн[а-я]*\s+срок|срок\s+гарант',text))
        if topic=='quality_acceptance':return bool(re.search(r'приемк',text) and re.search(r'качеств|количеств',text))
        if topic=='defect_claim':return bool(re.search(r'недостат|дефект',text) and re.search(r'претензи|акт',text))
        if topic=='price_inclusions':return bool(re.search(r'цен|стоим',text) and re.search(r'включ|упаков',text))
        if topic=='contract_term':return bool(re.search(r'договор\s+(?:действует|вступает|заключен)|продлева|отказаться\s+от\s+исполнения',text))
        if topic=='rent':return bool(re.search(r'арендная\s+плата\s+(?:состав|внос|уплач)|коммунальн.*оплач',text))
        if topic=='access':return bool(re.search(r'доступ|пропуск',text) and re.search(r'помещени|территор|круглосуточ|склад',text))
        if topic=='return_premises':return bool(re.search(r'возвращ|возврат',text) and re.search(r'помещени|аренд',text))
        if topic=='approval_general':return bool(re.search(r'генеральн.*директор',text) and re.search(r'одобрени|подлежат|согласуются',text))
        if topic=='approval_finance':return bool(re.search(r'финансов[а-я]*\s+директор|типов[а-я]*\s+форм',text))
        if topic=='approval_initiator':return 'инициатором' in text
        if topic=='originals_transfer':return bool('оригинал' in text and re.search(r'переда|юридическ',text))
        if topic=='foreign_counterparty':return bool(re.search(r'иностранн.*контраг|санкционн.*проверк|валютн.*контрол',text))
        if topic=='approval':return bool(re.search(r'согласуется|согласуются|одобрению|инициатором',text))
        if topic=='archive':
            if re.search(r'взять|выдач|четыр|на .*дн',norm(question)):return bool(re.search(r'выдач|срок.*выда',text))
            return bool(re.search(r'архив|хранени.*оригинал',text))
        return False
    return [c for c in _chunks(corpus,doc_id) if matches(c)]


def _scope_issues(registry,doc_ids,contract=None,clauses=None):
    issues=[]
    for issue in registry.issues:
        if issue.get('doc_id') not in doc_ids and (not contract or issue.get('contract_no')!=contract):continue
        target=issue.get('target_clause')
        if target and clauses is not None and target not in clauses:continue
        issues.append(issue)
    return issues


def _clarify_rules(issues,rules=()):
    return _result(intro='Не удалось однозначно проверить условие применения документа. Нужна проверка указанного условия по исходному тексту.',
                   status='clarify',missing=[i.get('reason','Условие применения документа') for i in issues][:4],rules=rules)


def _search(corpus,ctx,*,k,doc_ids):
    search=ctx.get('_retrieval_search') or corpus.search
    return search(ctx['query'],k=k,doc_ids=doc_ids)


def _contract_route(corpus,registry,ctx,number):
    profiles=[p for p in registry.profiles.values() if p.document_type=='contract' and number in p.contract_numbers]
    if not profiles:
        return _result(intro=f'Договор {number} не найден в предоставленной базе. Нужен текст этого договора.',status='abstain',missing=[f'Договор {number}'])
    if len(profiles)>1:
        return _result(intro='Найдено несколько документов с одним номером договора. Уточните стороны или редакцию.',status='clarify',missing=['Нужная редакция договора'])
    profile=profiles[0]
    topic=ctx.get('topic','generic')
    if topic=='payment' and re.search(r'договор\s+аренды',norm(corpus.documents[profile.doc_id].text[:300])):topic='rent'
    chunks=_topic_chunks(corpus,profile.doc_id,topic,ctx['text'])
    if not chunks:
        chunks=_search(corpus,ctx,k=5,doc_ids=[profile.doc_id])
        status='partial'
    else:status='answer'
    clauses={c.clause for c in chunks}
    issues=_scope_issues(registry,{profile.doc_id},number,clauses)
    if issues:return _clarify_rules(issues)
    rules=[]
    for clause in sorted(clauses):
        amendments=[a for a in registry.amendments if a.contract_no==number and a.target_clause==clause]
        applicable=[]
        unknown=[]
        for amendment in amendments:
            decision=evaluate_conditions(amendment.conditions,ctx)
            rules.append(_trace_rule(amendment,decision))
            if decision is True:applicable.append(amendment)
            elif decision is None:unknown.append(amendment)
        if unknown:
            fields={c.field for a in unknown for c in a.conditions if evaluate_condition(c,ctx) is None}
            need='Дата подписания спецификации' if 'specification_date' in fields else 'Точная дата события'
            return _result(intro='Для выбора применимого условия нужна точная дата соответствующего события. Дата поставки не заменяет дату подписания спецификации.',status='clarify',missing=[need],rules=rules)
        if len(applicable)>1:
            return _result(intro='К одному пункту применимы несколько изменений. По базе нельзя однозначно выбрать итоговую редакцию.',status='clarify',missing=['Порядок применения конкурирующих изменений'],rules=rules)
        if applicable:
            amendment=applicable[0]
            replacement=[c for c in _chunks(corpus,amendment.doc_id) if c.clause==clause]
            if not replacement:
                return _result(intro='Условие изменения найдено, но текст новой редакции пункта не выделен однозначно.',status='clarify',missing=['Текст заменяющего пункта'],rules=rules)
            chunks=[c for c in chunks if c.clause!=clause]+replacement
    if not chunks:
        return _result(intro='В выбранном договоре не найдено положения по вопросу.',status='abstain',missing=['Положение выбранного договора по вопросу'])
    return _result(chunks,'В применимых положениях договора указано:',status=status,rules=rules)


def _select_revisions(profiles,ctx):
    if not ctx.get('event_span'):
        open_ended=[p for p in profiles if not any(c.field=='event_date' and c.op in {'lt','lte'} for c in p.conditions)]
        if len(open_ended)==1:return open_ended,[]
        return [],['Укажите дату события и нужную редакцию регламента']
    decisions=[(p,evaluate_conditions(p.conditions,ctx)) for p in profiles]
    if any(result is None for _,result in decisions):return [],['Точная дата события: период пересекает границу действия редакции']
    active=[p for p,result in decisions if result is True]
    if len(active)>1:return [],['В базе несколько одновременно применимых редакций']
    return active,[]


def _regulation_route(corpus,registry,ctx):
    topic=ctx.get('topic')
    profiles=[p for p in registry.profiles.values() if p.document_type=='regulation']
    if not profiles:return None
    if len({p.family for p in profiles})>1:
        mentioned=[p for p in profiles if any(norm(e) in ctx['full'] for e in p.entities)]
        if mentioned and len({p.family for p in mentioned})==1:profiles=mentioned
        else:return _result(intro='В базе несколько регламентов. Уточните организацию или название регламента.',status='clarify',missing=['Организация или регламент'])
    issues=_scope_issues(registry,{p.doc_id for p in profiles})
    if issues:return _clarify_rules(issues)
    selected,missing=_select_revisions(profiles,ctx)
    if missing:return _result(intro='Нельзя однозначно выбрать редакцию по указанному периоду.',status='clarify',missing=missing)
    if not selected:return _result(intro='В базе нет подтверждённой редакции регламента на указанную дату.',status='abstain',missing=['Регламент на дату события'])
    chunks=[]
    rules=[_trace_rule(p,evaluate_conditions(p.conditions,ctx) if ctx.get('event_span') else None,
                       'event_date' if ctx.get('event_span') else 'latest_available_without_event_date')
           for p in profiles]
    for profile in selected:
        chunks+=_topic_chunks(corpus,profile.doc_id,topic,ctx['text'])
    if topic=='foreign_counterparty':
        families={p.family for p in selected}
        candidates=[p for p in registry.profiles.values() if p.document_type=='letter' and _topic_chunks(corpus,p.doc_id,topic)]
        unresolved=[p for p in candidates if p.family is None]
        if unresolved:
            issues=_scope_issues(registry,{p.doc_id for p in unresolved})
            return _clarify_rules(issues or [{'reason':'К какому регламенту относится дополнительное требование письма'}],rules)
        letters=[p for p in candidates if p.family in families]
        for profile in letters:
            issues=_scope_issues(registry,{profile.doc_id})
            if issues:return _clarify_rules(issues,rules)
            decision=evaluate_conditions(profile.conditions,ctx) if ctx.get('event_span') else True
            rules.append(_trace_rule(profile,decision if ctx.get('event_span') else None,
                                     'event_date' if ctx.get('event_span') else 'latest_available_without_event_date'))
            if decision is None:return _result(intro='Указанный период пересекает дату введения дополнительного требования. Уточните дату.',status='clarify',missing=['Точная дата события'],rules=rules)
            if decision:chunks+=_topic_chunks(corpus,profile.doc_id,topic)
    if not chunks:
        chunks=_search(corpus,ctx,k=5,doc_ids=[p.doc_id for p in selected])
        return _result(chunks,'Найденные положения выбранной редакции:',status='partial',rules=rules,forced=False)
    intro='Применимые положения регламента и дополнительных требований:' if ctx.get('event_span') else 'Дата события не указана. Положения последней доступной редакции и дополняющих документов:'
    return _result(chunks,intro,rules=rules)


def _law_docs(corpus,ctx):
    docs=[d for d in corpus.documents.values() if d.kind=='law']
    patterns={'law_probation':r'испытан|испытательн','law_fixed_term':r'срочн.*трудов','law_consent':r'согласие.*персональн|персональн.*согласи'}
    if ctx.get('topic') in patterns:docs=[d for d in docs if re.search(patterns[ctx['topic']],norm(d.text[:600]))]
    article=re.search(r'(?:стать[а-я]*|ст\.)\s*(\d+(?:\.\d+)?)',ctx['text'])
    if article:docs=[d for d in docs if re.search(r'(?:статья|ст\.)\s*'+re.escape(article[1])+r'(?!\d|\.\d)',d.text[:600],re.I)]
    code=re.search(r'\b(нк|тк|коап)\b',ctx['text'])
    if code:docs=[d for d in docs if re.search(r'\b'+code[1]+r'\b',norm(d.text[:600]))]
    return docs


def _law_route(corpus,ctx):
    docs=_law_docs(corpus,ctx)
    span=ctx.get('event_span')
    if not span:
        years=[int(y) for y in re.findall(r'\b(?:19|20)\d\d\b',ctx['text'])]
        if years:span=(date(min(years),1,1),date(max(years),12,31))
    snapshots=[date.fromisoformat(d.snapshot_date) for d in docs if d.snapshot_date]
    if span and snapshots:
        if any(span[0]<snapshot<=span[1] for snapshot in snapshots):
            return _result(intro='Период пересекает дату имеющегося снимка закона. Старых редакций в базе нет.',status='clarify',missing=['Точная дата события'])
        if any(span[1]<snapshot for snapshot in snapshots):
            return _result(intro='В базе есть только снимок статей от '+min(snapshots).strftime('%d.%m.%Y')+'. Подтвердить норму на более раннюю дату нельзя.',status='abstain',missing=['Редакция закона на дату события'])
    point=re.search(r'(?:пункт[а-я]*|п\.)\s*(\d+(?:\.\d+)*)',ctx['text'])
    if point and len(docs)==1:
        wanted=point[1]
        body=[c for c in _chunks(corpus,docs[0].id) if c.content_role=='body' and re.match(r'^\s*'+re.escape(wanted)+r'\.\s',c.text)]
        if not body:
            notes=[c for c in _chunks(corpus,docs[0].id) if c.content_role=='editorial' and re.search(r'\bпункт[а-я]*\s+'+re.escape(wanted)+r'(?!\d|\.\d)',norm(c.text))]
            return _result(notes[:1],f'В файле есть сведения о статье, но самого текста пункта нет. Воспроизвести пункт {wanted} по базе невозможно.',status='abstain',missing=[f'Полный текст пункта {wanted}'])
    if re.search(r'пособи',ctx['text']) and not any(re.search(r'пособи',norm(d.text)) for d in docs):
        return _result(intro='В предоставленных документах нет нормы, позволяющей определить запрошенное пособие.',status='abstain',missing=['Нормы расчёта пособия и исходные данные'])
    if not docs:return _result(intro='Нужная статья или нормативный источник не найдены в базе.',status='abstain',missing=['Нормативный документ по вопросу'])
    return _result(_search(corpus,ctx,k=4,doc_ids=[d.id for d in docs]),'Релевантные положения предоставленных статей:',status='partial',forced=False)


def route(corpus,ctx,registry=None,search=None):
    registry=registry or RuleRegistry(corpus)
    ctx=resolve_context(ctx,registry)
    if search is not None:ctx['_retrieval_search']=search
    t=ctx['text']
    if re.search(r'как\s+(?:скрыть|подделать|обойти|украсть)|помоги\s+(?:скрыть|подделать|обойти)',t):
        return _result(intro='Не могу помогать обходить закон или подделывать документы. Могу найти требования и законный порядок действий в базе.',status='abstain')
    if re.search(r'погода|напиши (?:стих|код)|рецепт (?:супа|борща)|столица|системн(?:ый|ого) промпт',t) and not re.search(r'договор|налог|аренд|согласован|оплат',t):
        return _result(intro='Я помогаю с юридическими вопросами по загруженным документам. Укажите договор, норму или рабочую ситуацию.',status='out_of_scope')
    if ctx.get('identity_conflict'):
        return _result(intro='Номер договора не согласуется с указанным контрагентом. Уточните договор и сторону.',status='clarify',missing=['Номер договора и соответствующий контрагент'])
    if ctx.get('ambiguous_entities'):
        return _result(intro='Указанное название или ИНН связано с несколькими договорами. Уточните номер договора или полное название контрагента.',status='clarify',missing=['Номер договора или точное название контрагента'])
    numbers=ctx.get('contracts',[])
    if len(numbers)>1:
        if not ctx.get('comparison') or set(numbers)!=set(ctx.get('explicit_contracts',[])):
            return _result(intro='Найдено несколько договоров. Уточните, какой нужно проверить.',status='clarify',missing=['Один номер договора'])
        parts=[_contract_route(corpus,registry,ctx,n) for n in numbers]
        if any(p is None for p in parts):return _result(intro='Уточните условие для сравнения договоров.',status='clarify',missing=['Сравниваемое условие'])
        incomplete=next((p for p in parts if p['status'] not in {'answer','partial'}),None)
        if incomplete:return incomplete
        return _result([c for p in parts for c in p['chunks']],'Сопоставление условий указанных договоров:',status='partial' if any(p['status']=='partial' for p in parts) else 'answer',rules=[r for p in parts for r in p['rules']])
    if numbers:return _contract_route(corpus,registry,ctx,numbers[0])
    topic=ctx.get('topic','generic')
    if topic=='archive':
        chunks=[c for p in registry.profiles.values() if p.document_type=='memo' for c in _topic_chunks(corpus,p.doc_id,topic,t)]
        if chunks:return _result(chunks,'Положения документов о хранении оригиналов:')
    if topic.startswith('approval') or topic in {'originals_transfer','foreign_counterparty'}:return _regulation_route(corpus,registry,ctx)
    if topic.startswith('law') or re.search(r'налог|\bнк\b|\bтк\b|\bкоап\b|пособи|испытательн|персональн.*данн|\bмрот\b',t):return _law_route(corpus,ctx)
    return None
