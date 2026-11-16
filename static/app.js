'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    documents: [], documentCache: new Map(), citations: [], history: [],
    tab: 'documents', currentDocument: null, mode: 'extractive', pending: false,
    requestId: 0, documentRequestId: 0, controller: null, lastQuestion: null, pendingTurn: null,
  };
  let historyStorage;
  try { historyStorage = window.localStorage; } catch (_) {
    historyStorage = { getItem() { throw new Error('Unavailable'); }, setItem() { throw new Error('Unavailable'); }, removeItem() {} };
  }
  const conversations = new PSBHistory.Store(historyStorage, message => {
    $('storage-message').textContent = message;
    $('storage-notice').hidden = !message;
  });
  const examples = {
    rent: 'Какая арендная плата по договору А-08/2024 и когда её нужно вносить?',
    payment: 'Верно ли, что по договору П-17/2024 оплата производится в течение 90 дней?',
    address: 'Какой адрес доставки применяется по договору П-17/2024 к партии со спецификацией от 02.10.2024?',
    approval: 'Нужно ли согласование финансового директора для обычного договора на 2 млн рублей в феврале 2025 года?',
    historical: 'Какую ставку НДС следовало применять в 2020 году? Есть ли в базе нужная редакция закона?',
  };
  const kindNames = {
    law: 'Законодательство', legal: 'Законодательство', legislation: 'Законодательство',
    company: 'Документ компании', corporate: 'Документ компании', contract: 'Договор', agreement: 'Договор',
    amendment: 'Допсоглашение', regulation: 'Регламент', policy: 'Регламент',
    letter: 'Письмо', memo: 'Служебная записка', internal: 'Документ компании',
  };
  const statusNames = {
    answer: 'Ответ с источниками', partial: 'Частичный ответ', clarify: 'Нужно уточнение',
    abstain: 'Недостаточно данных', out_of_scope: 'Вне базы знаний', error: 'Технический сбой',
  };
  const icons = {
    file: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3h10l4 4v14H5zM14 3v5h5M8 12h8M8 16h6"/></svg>',
    link: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m10 14 4-4M8 16l-2 2a4 4 0 0 1-6-6l5-5a4 4 0 0 1 6 0M16 8l2-2a4 4 0 0 1 6 6l-5 5a4 4 0 0 1-6 0" transform="translate(1 -1) scale(.9)"/></svg>',
    arrow: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M7 7h10v10"/></svg>',
    copy: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 8h12v13H8zM16 8V3H3v13h5"/></svg>',
    check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6"/></svg>',
  };

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  }
  // Only fixed, application-owned icon markup is inserted as HTML. All API data uses textContent.
  function icon(name, className = '') {
    const node = element('span', className);
    node.innerHTML = icons[name] || icons.file;
    return node;
  }
  function plural(number, forms) {
    const n = Math.abs(number) % 100;
    return forms[n > 10 && n < 20 ? 2 : n % 10 === 1 ? 0 : n % 10 >= 2 && n % 10 <= 4 ? 1 : 2];
  }
  function kindLabel(kind) { return kindNames[kind] || kind || 'Документ'; }
  function titleOf(doc) { return doc.title || doc.filename || doc.id || 'Документ'; }
  function formatDate(value) {
    if (!value) return '';
    const parts = value.split('-');
    return parts.length === 3 ? `${parts[2]}.${parts[1]}.${parts[0]}` : value;
  }
  function scrollToBottom() {
    requestAnimationFrame(() => { $('conversation-scroll').scrollTop = $('conversation-scroll').scrollHeight; });
  }
  function toast(message) {
    $('toast').textContent = message;
    $('toast').hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => { $('toast').hidden = true; }, 2600);
  }
  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      toast('Скопировано');
    } catch (_) {
      toast('Не удалось скопировать. Выделите текст вручную.');
    }
  }
  async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    let payload;
    try { payload = await response.json(); } catch (_) { throw new Error('Сервер вернул неожиданный ответ. Попробуйте ещё раз.'); }
    if (!response.ok) {
      const detail = typeof payload.detail === 'string' ? payload.detail : typeof payload.error === 'string' ? payload.error : '';
      throw new Error(detail || `Не удалось выполнить запрос (${response.status}). Попробуйте ещё раз.`);
    }
    return payload;
  }

  function updateMode(health) {
    state.mode = health.mode === 'llm' ? 'llm' : 'extractive';
    $('mode-label').textContent = state.mode === 'llm' ? 'API настроен' : 'Поиск по документам';
    $('mode-status').title = state.mode === 'llm' && health.model ? `Модель: ${health.model}` : 'Поиск и цитирование положений учебной базы';
    $('mode-status').querySelector('.status-dot').className = 'status-dot';
    $('composer-footnote').textContent = state.mode === 'extractive'
      ? 'Поиск по документам. Для свободных ответов подключите модель.'
      : 'Учебный корпус. Проверяйте применимость положений по источникам.';
    const count = Number(health.documents);
    if (Number.isFinite(count)) updateDocumentCount(count);
    $('connection-notice').hidden = true;
  }
  function updateDocumentCount(count) {
    $('sidebar-document-count').textContent = String(count);
    $('sidebar-document-count').nextElementSibling.textContent = `${plural(count, ['документ', 'документа', 'документов'])} в базе`;
    $('source-count').textContent = String(count);
    $('source-count').title = `${count} ${plural(count, ['документ', 'документа', 'документов'])} в базе`;
  }
  async function loadHealth() {
    try { updateMode(await fetchJson('/api/health')); }
    catch (_) {
      $('mode-label').textContent = 'Нет подключения';
      $('mode-status').querySelector('.status-dot').className = 'status-dot error';
      $('connection-message').textContent = 'Не удалось подключиться к базе. Проверьте, что сервер запущен.';
      $('connection-notice').hidden = false;
    }
  }
  async function loadDocuments() {
    try {
      const data = await fetchJson('/api/documents');
      state.documents = Array.isArray(data.documents) ? data.documents : [];
      updateDocumentCount(state.documents.length);
      const selected = $('document-kind').value;
      $('document-kind').replaceChildren(new Option('Все типы документов', 'all'));
      [...new Set(state.documents.map(d => d.kind).filter(Boolean))].sort().forEach(kind => {
        $('document-kind').add(new Option(kindLabel(kind), kind));
      });
      if ([...$('document-kind').options].some(option => option.value === selected)) $('document-kind').value = selected;
      renderDocuments();
    } catch (_) {
      const error = element('div', 'panel-error', 'Не удалось загрузить список документов.');
      const retry = element('button', '', 'Загрузить ещё раз');
      retry.type = 'button';
      retry.addEventListener('click', loadDocuments);
      error.append(retry);
      $('document-list').replaceChildren(error);
    }
  }
  function renderDocuments() {
    const query = $('document-search').value.trim().toLocaleLowerCase('ru');
    const kind = $('document-kind').value;
    const docs = state.documents.filter(doc =>
      (kind === 'all' || doc.kind === kind) &&
      `${titleOf(doc)} ${doc.filename || ''}`.toLocaleLowerCase('ru').includes(query)
    );
    $('document-list').replaceChildren();
    if (!docs.length) {
      $('document-list').append(element('div', 'empty-documents', state.documents.length
        ? 'Документы не найдены. Попробуйте другой номер или название.'
        : 'В базе пока нет документов.'));
      return;
    }
    const fragment = document.createDocumentFragment();
    docs.forEach(doc => {
      const button = element('button', 'document-list-item');
      button.type = 'button';
      button.setAttribute('aria-label', `Открыть документ: ${titleOf(doc)}`);
      const copy = element('span', 'document-item-copy');
      copy.append(element('strong', '', titleOf(doc)), element('small', '', kindLabel(doc.kind)));
      button.append(icon('file', 'document-list-icon'), copy, element('span', 'document-list-arrow', '›'));
      button.addEventListener('click', () => openDocument(doc.id));
      fragment.append(button);
    });
    $('document-list').append(fragment);
  }
  function setTab(tab, focus = false) {
    state.tab = tab;
    state.documentRequestId += 1;
    $('document-view').hidden = true;
    for (const name of ['evidence', 'documents']) {
      const active = name === tab;
      $(`tab-${name}`).setAttribute('aria-selected', String(active));
      $(`tab-${name}`).tabIndex = active ? 0 : -1;
      $(`${name}-panel`).hidden = !active;
    }
    if (focus) $(`tab-${tab}`).focus();
  }
  function toggleSourcePanel(open) {
    const mobile = window.matchMedia('(max-width: 760px)').matches;
    const drawerOpen = mobile && open;
    $('source-panel').classList.toggle('is-open', drawerOpen);
    $('source-backdrop').hidden = !drawerOpen;
    $('mobile-sources').setAttribute('aria-expanded', String(drawerOpen));
    if (drawerOpen) $('close-source-panel').focus();
    if (!open && mobile) $('mobile-sources').focus();
  }
  function setCitations(citations, status = 'answer') {
    state.citations = Array.isArray(citations) ? citations : [];
    $('evidence-count').textContent = String(state.citations.length);
    $('evidence-count').hidden = !state.citations.length;
    const container = $('evidence-content');
    container.replaceChildren();
    if (!state.citations.length) {
      const empty = element('div', 'evidence-empty');
      empty.append(element('div', 'evidence-empty-graphic'));
      empty.append(element('h3', '', state.history.length ? 'В этом ответе нет цитат' : 'За каждым выводом — источник'));
      empty.append(element('p', '', state.history.length
        ? status === 'clarify' ? 'Уточните вопрос, чтобы найти подходящие положения.' : 'Проверьте, каких сведений не хватает, или уточните вопрос.'
        : 'Задайте вопрос. Здесь появятся положения документов, на которых основан ответ.'));
      container.append(empty);
      return;
    }
    container.append(element('p', 'evidence-intro', 'Откройте цитату, чтобы проверить её в полном тексте документа.'));
    state.citations.forEach((citation, index) => {
      const button = element('button', 'evidence-card');
      button.type = 'button';
      button.setAttribute('aria-label', `Источник ${index + 1}: ${citation.title || 'Документ'}, ${citation.clause || 'фрагмент'}`);
      const heading = element('span', 'evidence-card-heading');
      heading.append(element('span', 'citation-index', index + 1), element('strong', '', citation.title || 'Документ'));
      button.append(heading);
      if (citation.clause) button.append(element('span', 'evidence-clause', citation.clause));
      button.append(element('span', 'evidence-quote', citation.quote || 'Откройте документ для просмотра положения.'));
      const footer = element('span', 'evidence-card-footer');
      footer.append(element('span', '', 'Открыть в документе'), icon('arrow'));
      button.append(footer);
      button.addEventListener('click', () => openDocument(citation.doc_id, citation));
      container.append(button);
    });
  }

  async function openDocument(id, citation = null) {
    if (id === undefined || id === null) { toast('У источника не указан документ.'); return; }
    const request = ++state.documentRequestId;
    state.currentDocument = null;
    const metadata = state.documents.find(doc => String(doc.id) === String(id));
    $('document-view-title').textContent = citation?.title || (metadata ? titleOf(metadata) : 'Документ');
    $('document-view-type').textContent = metadata ? kindLabel(metadata.kind) : 'Источник';
    $('document-view-description').textContent = citation?.clause ? `${citation.clause} · Подтверждающий фрагмент выделен` : 'Полный исходный текст';
    $('document-text').replaceChildren();
    $('document-loading').hidden = false;
    $('document-text-container').hidden = true;
    $('evidence-panel').hidden = true;
    $('documents-panel').hidden = true;
    $('document-view').hidden = false;
    $('copy-document').disabled = true;
    toggleSourcePanel(true);
    try {
      let doc = state.documentCache.get(String(id));
      if (!doc) {
        doc = await fetchJson(`/api/documents/${encodeURIComponent(id)}`);
        state.documentCache.set(String(id), doc);
      }
      if (request !== state.documentRequestId) return;
      state.currentDocument = doc;
      $('document-view-title').textContent = titleOf(doc);
      $('copy-document').disabled = false;
      renderDocumentText(doc, citation);
    } catch (error) {
      if (request !== state.documentRequestId) return;
      $('document-text').classList.add('document-error');
      $('document-text').textContent = 'Не удалось открыть документ. Вернитесь к списку и попробуйте ещё раз.';
      $('document-view-description').textContent = 'Источник временно недоступен';
    } finally {
      if (request === state.documentRequestId) {
        $('document-loading').hidden = true;
        $('document-text-container').hidden = false;
      }
    }
  }
  function renderDocumentText(doc, citation) {
    const source = String(doc.text || '');
    const chars = Array.from(source); // API offsets refer to Python Unicode code points, not UTF-16 units.
    const container = $('document-text');
    container.classList.remove('document-error');
    container.replaceChildren();
    let start = citation ? Number(citation.start) : NaN;
    let end = citation ? Number(citation.end) : NaN;
    let exact = Number.isInteger(start) && Number.isInteger(end) && start >= 0 && end > start && end <= chars.length;
    if (exact && citation.quote) exact = chars.slice(start, end).join('') === citation.quote;
    if (!exact && citation?.quote) {
      const index = source.indexOf(citation.quote);
      if (index >= 0) {
        start = Array.from(source.slice(0, index)).length;
        end = start + Array.from(citation.quote).length;
        exact = true;
      }
    }
    if (exact) {
      container.append(document.createTextNode(chars.slice(0, start).join('')));
      const highlight = element('mark', 'source-highlight', chars.slice(start, end).join(''));
      highlight.id = 'source-highlight';
      container.append(highlight, document.createTextNode(chars.slice(end).join('')));
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const scroll = $('document-text-container');
        const relativeTop = highlight.getBoundingClientRect().top - scroll.getBoundingClientRect().top;
        scroll.scrollTop = Math.max(0, scroll.scrollTop + relativeTop - 60);
      }));
    } else {
      container.textContent = source || 'В документе нет текстового содержимого.';
      $('document-text-container').scrollTop = 0;
      if (citation) $('document-view-description').textContent = 'Полный текст. Точное выделение цитаты недоступно — проверьте указанный пункт.';
    }
  }

  function appendUser(question, date) {
    $('welcome').hidden = true;
    const article = element('article', 'message user');
    article.setAttribute('aria-label', 'Ваш вопрос');
    article.append(element('div', 'message-meta', 'Вы'), element('div', 'user-message-bubble', question));
    if (date) article.append(element('div', 'event-date-note', `Дата события: ${formatDate(date)}`));
    $('messages').append(article);
    scrollToBottom();
    return article;
  }
  function createAssistantHeading(status, mode) {
    const heading = element('div', 'assistant-heading');
    heading.append(icon('check', 'assistant-avatar'), element('strong', '', mode === 'extractive' ? 'По документам' : 'Юридический ассистент'));
    heading.append(element('span', `answer-status ${status}`, statusNames[status] || 'Ответ'));
    return heading;
  }
  function appendFormattedText(parent, text, citations) {
    const sourceIndex = new Map(citations.map((citation, index) => [citation.source_id, index]).filter(([sid]) => sid));
    String(text || '').split(/\n\s*\n/).forEach(paragraph => {
      if (!paragraph.trim()) return;
      const p = element('p');
      const tokens = paragraph.split(/(\*\*[^*]+\*\*|\[(?:S\d+(?:\s*[,;]\s*S\d+)*|\d+)\])/g);
      tokens.forEach(token => {
        if (/^\*\*[^*]+\*\*$/.test(token)) p.append(element('strong', '', token.slice(2, -2)));
        else if (/^\[(?:S\d+(?:\s*[,;]\s*S\d+)*|\d+)\]$/.test(token)) {
          const references = token.slice(1, -1).split(/\s*[,;]\s*/);
          references.forEach(ref => {
            const index = ref.startsWith('S') ? sourceIndex.get(ref) : Number(ref) - 1;
            if (citations[index]) {
              const button = element('button', 'inline-citation', `[${index + 1}]`);
              button.type = 'button';
              button.setAttribute('aria-label', `Открыть источник ${index + 1}`);
              button.addEventListener('click', () => {
                setCitations(citations); setTab('evidence'); openDocument(citations[index].doc_id, citations[index]);
              });
              p.append(button);
            } else p.append(document.createTextNode(`[${ref}]`));
          });
        } else p.append(document.createTextNode(token));
      });
      parent.append(p);
    });
  }
  function appendAnswer(data) {
    const citations = Array.isArray(data.citations) ? data.citations : [];
    const article = element('article', 'message assistant');
    article.setAttribute('aria-label', 'Ответ ассистента');
    article.append(createAssistantHeading(data.status || 'answer', data.mode || state.mode));
    const body = element('div', 'answer-body');
    appendFormattedText(body, data.answer || 'В ответе не оказалось текста.', citations);
    article.append(body);
    const missing = Array.isArray(data.missing_facts) ? data.missing_facts : [];
    if (missing.length) {
      const box = element('div', 'missing-facts');
      box.append(element('strong', '', 'Что ещё нужно уточнить'));
      const list = element('ul');
      missing.forEach(fact => list.append(element('li', '', typeof fact === 'string' ? fact : JSON.stringify(fact))));
      box.append(list); article.append(box);
    }
    if (citations.length) {
      const sources = element('div', 'answer-sources');
      const title = element('p', 'answer-sources-title');
      title.append(icon('link'), document.createTextNode('Проверить основания'));
      sources.append(title);
      citations.forEach((citation, index) => {
        const button = element('button', 'answer-source-button');
        button.type = 'button';
        button.setAttribute('aria-label', `Проверить источник ${index + 1}: ${citation.title || 'Документ'}`);
        const name = element('span', 'answer-source-name');
        name.append(document.createTextNode(citation.title || 'Документ'));
        if (citation.clause) { name.append(document.createTextNode(' · '), element('strong', '', citation.clause)); }
        button.append(element('span', 'citation-index', index + 1), name, element('span', 'source-out-icon', '↗'));
        button.addEventListener('click', () => { setCitations(citations, data.status); setTab('evidence'); openDocument(citation.doc_id, citation); });
        sources.append(button);
      });
      article.append(sources);
    }
    const warnings = Array.isArray(data.warnings) ? data.warnings : [];
    if (warnings.length) {
      const details = element('details', 'answer-warnings');
      details.append(element('summary', '', `Примечания к ответу · ${warnings.length}`));
      warnings.forEach(warning => details.append(element('p', '', typeof warning === 'string' ? warning : JSON.stringify(warning))));
      article.append(details);
    }
    const footer = element('div', 'answer-footer');
    const copy = element('button', 'copy-answer');
    copy.type = 'button'; copy.setAttribute('aria-label', 'Скопировать ответ');
    copy.append(icon('copy'), document.createTextNode('Копировать'));
    copy.addEventListener('click', () => copyText(String(data.answer || '')));
    footer.append(copy);
    const selfCheck = data.self_check || data.trace?.generation?.self_check;
    if (selfCheck?.completed === true) {
      footer.append(element('span', 'self-check completed', 'Самопроверка выполнена'));
    } else if (data.mode === 'llm' && selfCheck?.enabled === true) {
      footer.append(element('span', 'self-check unavailable', 'Самопроверка недоступна'));
    }
    if (data.trace && Number.isFinite(Number(data.trace.total_ms))) {
      const seconds = Number(data.trace.total_ms) / 1000;
      footer.append(element('span', '', `${seconds.toLocaleString('ru-RU', { maximumFractionDigits: 1 })} сек`));
    }
    article.append(footer);
    $('messages').append(article);
    setCitations(citations, data.status);
    setTab('evidence');
    scrollToBottom();
  }
  function setPending(pending) {
    state.pending = pending;
    $('send-button').disabled = pending;
    $('send-button').setAttribute('aria-label', pending ? 'Ожидаем ответ' : 'Отправить вопрос');
    $('send-button').querySelector('span').textContent = pending ? 'Ищем…' : 'Спросить';
    $('chat-form').setAttribute('aria-busy', String(pending));
    document.querySelectorAll('.retry-button').forEach(button => { button.disabled = pending; });
  }
  function appendFailure(turn) {
    const failure = element('div', 'message error-message');
    failure.setAttribute('role', 'alert');
    failure.append(element('p', '', turn.error || PSBHistory.interrupted));
    const retry = element('button', 'retry-button', 'Повторить вопрос');
    retry.type = 'button'; retry.disabled = state.pending;
    retry.addEventListener('click', () => submitQuestion(turn.question, turn.date, turn.id));
    failure.append(retry); $('messages').append(failure);
  }
  function renderConversation() {
    const chat = conversations.active();
    state.history = PSBHistory.context(chat);
    $('messages').replaceChildren();
    $('welcome').hidden = Boolean(chat?.turns.length);
    setCitations([]); setTab('documents');
    for (const turn of chat?.turns || []) {
      appendUser(turn.question, turn.date);
      if (turn.status === 'complete') appendAnswer(turn.answerData);
      else if (turn.status !== 'pending') appendFailure(turn);
    }
    scrollToBottom();
  }
  function cancelPending() {
    state.requestId += 1;
    state.controller?.abort(); state.controller = null;
    if (state.pendingTurn) {
      conversations.fail(state.pendingTurn.chatId, state.pendingTurn.turnId, PSBHistory.interrupted, 'interrupted');
      state.pendingTurn = null;
    }
    setPending(false);
  }
  async function submitQuestion(question, date = null, retryId = null) {
    if (state.pending || !question.trim()) return;
    question = question.trim();
    const chat = conversations.active();
    const history = PSBHistory.context(chat, retryId);
    const dialogueState = PSBHistory.dialogueState(chat, retryId);
    const request = ++state.requestId;
    const controller = new AbortController();
    state.controller = controller;
    state.lastQuestion = { question, date };
    // Save before starting the request, including retries after a reload.
    const turn = conversations.begin(question, date, retryId);
    state.pendingTurn = { chatId: chat.id, turnId: turn.id };
    $('question').value = ''; resizeTextarea();
    $('date-popover').hidden = true;
    $('date-toggle').setAttribute('aria-expanded', 'false');
    setPending(true); renderConversation();
    const loading = element('div', 'message loading-message');
    loading.setAttribute('role', 'status');
    const dots = element('span', 'loading-dots');
    for (let i = 0; i < 3; i++) dots.append(element('i'));
    loading.append(dots, document.createTextNode(state.mode === 'llm' ? 'Уточняем смысл вопроса, ищем основания и проверяем ответ…' : 'Проверяем положения документов…'));
    $('messages').append(loading); scrollToBottom();
    const timer = setTimeout(() => controller.abort(), 180000);
    try {
      const result = await fetchJson('/api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, history, event_date: date || null, dialogue_state: dialogueState }), signal: controller.signal,
      });
      if (request !== state.requestId) return;
      if (typeof result.answer !== 'string' || !result.answer.trim()) throw new Error('В ответе не оказалось текста. Попробуйте ещё раз.');
      conversations.complete(chat.id, turn.id, result);
    } catch (error) {
      if (request !== state.requestId) return;
      const message = error.name === 'AbortError'
        ? 'Ответ занимает слишком много времени. Вопрос сохранён — попробуйте повторить его.'
        : error instanceof TypeError ? 'Связь с сервером прервалась. Проверьте подключение и повторите запрос.' : error.message;
      conversations.fail(chat.id, turn.id, message);
    } finally {
      clearTimeout(timer);
      if (request === state.requestId) {
        setPending(false); state.controller = null; state.pendingTurn = null;
        renderConversation(); renderHistory();
      }
    }
  }
  function resetConversation() {
    cancelPending();
    conversations.newChat(); state.lastQuestion = null;
    $('question').value = ''; $('event-date').value = ''; updateDate(); resizeTextarea();
    renderConversation(); renderHistory();
    toggleSourcePanel(false); $('conversation-scroll').scrollTop = 0;
    $('question').focus(); toast('Начат новый диалог. Предыдущий сохранён в истории.');
  }
  function renderHistory() {
    const list = $('history-list'); list.replaceChildren();
    const chats = conversations.data.chats.filter(chat => chat.turns.length).slice().sort((a, b) => b.updatedAt - a.updatedAt);
    $('clear-history').disabled = !chats.length && !conversations.readFailed;
    if (!chats.length) list.append(element('p', 'history-empty', 'Здесь появятся ваши диалоги после первого вопроса.'));
    for (const chat of chats) {
      const row = element('div', `history-row${chat.id === conversations.data.activeId ? ' active' : ''}`);
      const open = element('button', 'history-chat'); open.type = 'button';
      const title = chat.turns[0].question.replace(/\s+/g, ' ');
      open.append(element('strong', '', title.length > 95 ? title.slice(0, 94) + '…' : title));
      const date = new Date(chat.updatedAt).toLocaleString('ru-RU', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
      open.append(element('span', '', `${date} · ${chat.turns.length} ${plural(chat.turns.length, ['вопрос', 'вопроса', 'вопросов'])}${chat.id === conversations.data.activeId ? ' · открыт' : ''}`));
      open.addEventListener('click', () => {
        cancelPending(); conversations.select(chat.id); renderConversation();
        $('question').value = ''; $('event-date').value = ''; updateDate(); resizeTextarea();
        $('history-dialog').close(); $('question').focus();
      });
      const remove = element('button', 'history-remove', '×'); remove.type = 'button';
      remove.setAttribute('aria-label', `Удалить диалог: ${title}`);
      remove.addEventListener('click', () => {
        if (chat.id === conversations.data.activeId) cancelPending();
        conversations.remove(chat.id); renderHistory(); renderConversation(); toast('Диалог удалён');
      });
      row.append(open, remove); list.append(row);
    }
  }
  function openHistory() {
    renderHistory(); $('history-delete-confirm').hidden = true; $('history-dialog').showModal();
  }
  function resizeTextarea() {
    $('question').style.height = 'auto';
    $('question').style.height = `${Math.min(160, Math.max(45, $('question').scrollHeight))}px`;
  }
  function updateDate() {
    const value = $('event-date').value;
    $('date-label').textContent = value ? formatDate(value) : 'Дата события';
    $('date-toggle').classList.toggle('has-date', Boolean(value));
    $('date-toggle').querySelector('.optional-label').hidden = Boolean(value);
  }

  $('chat-form').addEventListener('submit', (event) => {
    event.preventDefault(); submitQuestion($('question').value, $('event-date').value || null);
  });
  $('question').addEventListener('input', resizeTextarea);
  $('question').addEventListener('keydown', event => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); $('chat-form').requestSubmit(); }
  });
  document.querySelectorAll('[data-example]').forEach(button => {
    button.addEventListener('click', () => {
      $('question').value = examples[button.dataset.example] || '';
      resizeTextarea(); $('question').focus();
    });
  });
  $('new-chat').addEventListener('click', resetConversation);
  $('compact-new-chat').addEventListener('click', resetConversation);
  $('open-history').addEventListener('click', openHistory);
  $('compact-history').addEventListener('click', openHistory);
  $('close-history').addEventListener('click', () => $('history-dialog').close());
  $('history-new-chat').addEventListener('click', () => { $('history-dialog').close(); resetConversation(); });
  $('clear-history').addEventListener('click', () => { $('history-delete-confirm').hidden = false; $('cancel-clear-history').focus(); });
  $('cancel-clear-history').addEventListener('click', () => { $('history-delete-confirm').hidden = true; $('clear-history').focus(); });
  $('confirm-clear-history').addEventListener('click', () => {
    cancelPending(); conversations.clear(); renderHistory(); renderConversation();
    $('history-delete-confirm').hidden = true; toast('История удалена из этого браузера');
  });
  $('retry-storage').addEventListener('click', () => { if (conversations.save()) toast('История сохранена'); });
  window.addEventListener('storage', event => {
    if (event.key !== PSBHistory.KEY && event.key !== null) return;
    conversations.refresh();
    if (state.pendingTurn && !conversations.chat(state.pendingTurn.chatId)) cancelPending();
    if (!conversations.active()) conversations.newChat();
    if (!state.pending) { renderConversation(); renderHistory(); }
  });
  $('open-library').addEventListener('click', () => { setTab('documents'); toggleSourcePanel(true); $('document-search').focus(); });
  $('document-search').addEventListener('input', renderDocuments);
  $('document-kind').addEventListener('change', renderDocuments);
  $('tab-evidence').addEventListener('click', () => setTab('evidence'));
  $('tab-documents').addEventListener('click', () => setTab('documents'));
  document.querySelector('.source-tabs').addEventListener('keydown', event => {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      setTab(event.key === 'Home' ? 'evidence' : event.key === 'End' ? 'documents' : state.tab === 'documents' ? 'evidence' : 'documents', true);
    }
  });
  $('document-back').addEventListener('click', () => setTab(state.tab));
  $('copy-document').addEventListener('click', () => { if (state.currentDocument) copyText(state.currentDocument.text || ''); });
  $('mobile-sources').addEventListener('click', () => toggleSourcePanel(!$('source-panel').classList.contains('is-open')));
  $('close-source-panel').addEventListener('click', () => toggleSourcePanel(false));
  $('source-backdrop').addEventListener('click', () => toggleSourcePanel(false));
  $('date-toggle').addEventListener('click', () => {
    const open = $('date-popover').hidden;
    $('date-popover').hidden = !open; $('date-toggle').setAttribute('aria-expanded', String(open));
    if (open) $('event-date').focus();
  });
  $('event-date').addEventListener('change', updateDate);
  $('clear-date').addEventListener('click', () => { $('event-date').value = ''; updateDate(); });
  document.addEventListener('click', event => {
    if (!event.target.closest('.date-control')) { $('date-popover').hidden = true; $('date-toggle').setAttribute('aria-expanded', 'false'); }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      $('date-popover').hidden = true; $('date-toggle').setAttribute('aria-expanded', 'false');
      if ($('source-panel').classList.contains('is-open')) toggleSourcePanel(false);
    }
    if (event.key === 'Tab' && window.matchMedia('(max-width: 760px)').matches && $('source-panel').classList.contains('is-open')) {
      const focusable = [...$('source-panel').querySelectorAll('button:not([disabled]), input, select, [tabindex="0"]')].filter(node => node.getClientRects().length > 0);
      const first = focusable[0], last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }
  });
  $('retry-connection').addEventListener('click', () => { loadHealth(); loadDocuments(); });
  window.addEventListener('resize', () => {
    if (!window.matchMedia('(max-width: 760px)').matches) toggleSourcePanel(false);
    else if ($('source-panel').classList.contains('is-open')) $('source-backdrop').hidden = false;
  });
  renderConversation();
  Promise.allSettled([loadHealth(), loadDocuments()]);
})();
