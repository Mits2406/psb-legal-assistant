'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    documents: [], documentCache: new Map(), citations: [], history: [],
    tab: 'documents', currentDocument: null, mode: 'extractive', pending: false,
    requestId: 0, documentRequestId: 0, controller: null, lastQuestion: null, pendingTurn: null,
    user: null, csrfToken: '', aclRevision: null, historyScope: '', authEpoch: 0,
    sessionRequestId: 0, authBusy: false, historyLoaded: false, manualLogin: false,
  };
  // Document answers are held in memory only. The authenticated server owns
  // durable history; another account must never read an earlier browser cache.
  function memoryHistory(raw = null) {
    let value = raw;
    return { getItem() { return value; }, setItem(_key, next) { value = next; }, removeItem() { value = null; } };
  }
  function makeHistory(raw = null) {
    return new PSBHistory.Store(memoryHistory(raw), message => {
      $('storage-message').textContent = message;
      $('storage-notice').hidden = !message;
    });
  }
  let conversations = makeHistory();
  try {
    for (const key of Object.keys(window.localStorage)) {
      if (key === PSBHistory.KEY || key.startsWith('psb.auth.history.')) window.localStorage.removeItem(key);
    }
  } catch (_) { /* No persistent client history is used by this version. */ }
  const examples = {
    rent: 'Какие условия арендной платы содержатся в доступных мне документах?',
    payment: 'Какие сроки оплаты предусмотрены в доступных мне договорах?',
    address: 'Как определить адрес доставки по доступным мне документам?',
    approval: 'Кто согласует договор и какие условия согласования указаны в доступных мне документах?',
    historical: 'Есть ли в доступных мне документах редакция закона, действовавшая на дату события?',
    generic: 'Найди условия оплаты в доступных мне документах.',
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
  function staleResponse() {
    const error = new Error('Запрос относится к предыдущему сеансу.');
    error.name = 'StaleSessionError';
    return error;
  }
  async function fetchJson(url, options = {}, sessionRequest = false) {
    const epoch = state.authEpoch;
    const headers = { ...options.headers };
    if ((options.method || 'GET').toUpperCase() !== 'GET' && state.csrfToken) headers['X-CSRF-Token'] = state.csrfToken;
    const response = await fetch(url, { ...options, headers, credentials: 'same-origin' });
    let payload;
    try { payload = await response.json(); } catch (_) { throw new Error('Сервер вернул неожиданный ответ. Попробуйте ещё раз.'); }
    if (!sessionRequest && epoch !== state.authEpoch) throw staleResponse();
    if (!response.ok) {
      const detail = typeof payload.detail === 'string' ? payload.detail : typeof payload.error === 'string' ? payload.error : '';
      if (response.status === 401 && !sessionRequest) signedOut('Сеанс завершён. Войдите снова.');
      if (response.status === 409 && !sessionRequest) {
        await refreshSession(true);
        toast('Права доступа изменились. Начните новый диалог.');
      }
      const error = new Error(detail || `Не удалось выполнить запрос (${response.status}). Попробуйте ещё раз.`);
      error.status = response.status;
      throw error;
    }
    return payload;
  }
  function clearWorkspace() {
    cancelPending();
    state.authEpoch += 1; state.documentRequestId += 1;
    state.documents = []; state.documentCache.clear(); state.currentDocument = null;
    state.history = []; state.citations = []; state.lastQuestion = null; state.historyLoaded = false;
    conversations = makeHistory();
    $('messages').replaceChildren(); $('document-list').replaceChildren(); $('document-text').replaceChildren();
    $('history-list').replaceChildren(); $('document-view-title').textContent = '';
    $('document-view-description').textContent = ''; $('document-view-type').textContent = '';
    $('document-access').hidden = true; $('document-access-controls').hidden = true;
    $('document-access-message').hidden = true; $('question').value = ''; $('event-date').value = '';
    $('storage-notice').hidden = true; $('toast').hidden = true; clearTimeout(toast.timer);
    $('source-panel').classList.remove('is-open'); $('source-backdrop').hidden = true;
    $('mobile-sources').setAttribute('aria-expanded', 'false');
    $('document-search').value = ''; $('document-kind').value = 'all';
    $('example-grid').hidden = true; $('historical-example').hidden = true; $('generic-example').hidden = true;
    if ($('history-dialog').open) $('history-dialog').close();
    setCitations([]); setTab('documents'); updateDocumentCount(0); updateDate();
  }
  function signedOut(message = '') {
    clearWorkspace();
    state.user = null; state.csrfToken = ''; state.aclRevision = null; state.historyScope = '';
    $('workspace').hidden = true; $('account-controls').hidden = true; $('mobile-sources').hidden = true;
    $('login-screen').hidden = false; $('login-error').textContent = message; $('login-error').hidden = !message;
    $('account-label').textContent = ''; $('login-password').value = '';
  }
  function validSession(payload) {
    return payload?.authenticated === true && typeof payload.user?.username === 'string'
      && [1, 2].includes(payload.user.level) && Number.isInteger(payload.acl_revision)
      && typeof payload.csrf_token === 'string' && payload.csrf_token.length > 0
      && typeof payload.history_scope === 'string' && payload.history_scope.length > 0;
  }
  async function loadHistory() {
    const epoch = state.authEpoch;
    const payload = await fetchJson('/api/history');
    if (epoch !== state.authEpoch || !state.user) return;
    const checked = PSBHistory.decode(JSON.stringify(payload.history));
    conversations = makeHistory(JSON.stringify(checked));
    state.historyLoaded = true;
    renderConversation(); renderHistory();
  }
  async function acceptSession(payload, forceReset = false) {
    if (!validSession(payload)) {
      // Session polling must not erase a password while a signed-out user types.
      if (state.user) signedOut('Сеанс завершён. Войдите снова.');
      return;
    }
    const changed = forceReset || state.historyScope !== payload.history_scope
      || state.user?.username !== payload.user.username || state.user?.level !== payload.user.level
      || state.aclRevision !== payload.acl_revision;
    const aclChanged = Boolean(state.user) && state.aclRevision !== payload.acl_revision;
    if (changed) clearWorkspace();
    state.user = { username: payload.user.username, level: payload.user.level };
    state.csrfToken = payload.csrf_token; state.aclRevision = payload.acl_revision; state.historyScope = payload.history_scope;
    $('account-label').textContent = `${state.user.username} · Уровень ${state.user.level}`;
    $('example-grid').hidden = state.user.level !== 2;
    $('historical-example').hidden = state.user.level !== 2;
    $('generic-example').hidden = state.user.level !== 1;
    $('example-hint').textContent = state.user.level === 2 ? 'Выберите пример' : 'Только доступные вам документы';
    $('account-controls').hidden = false; $('mobile-sources').hidden = false;
    $('login-screen').hidden = true; $('workspace').hidden = false; $('login-error').hidden = true;
    if (changed) {
      const epoch = state.authEpoch;
      try { await loadHistory(); }
      catch (error) {
        if (epoch !== state.authEpoch) return;
        signedOut('Не удалось загрузить историю и проверить права. Войдите снова.');
        return;
      }
      if (epoch !== state.authEpoch) return;
      await Promise.allSettled([loadHealth(), loadDocuments()]);
      if (aclChanged && epoch === state.authEpoch) toast('Права доступа изменились. Начните новый диалог.');
    }
  }
  async function refreshSession(forceReset = false) {
    if (state.authBusy || state.manualLogin) return;
    const request = ++state.sessionRequestId;
    try {
      const payload = await fetchJson('/api/session', {}, true);
      if (request !== state.sessionRequestId) return;
      await acceptSession(payload, forceReset);
    } catch (_) {
      if (request !== state.sessionRequestId) return;
      const message = 'Не удалось проверить сеанс. Проверьте подключение и войдите снова.';
      if (state.user) signedOut(message);
      else { $('login-error').textContent = message; $('login-error').hidden = false; }
    }
  }
  async function login(event) {
    event.preventDefault();
    if (state.authBusy) return;
    state.authBusy = true; state.sessionRequestId += 1;
    $('login-submit').disabled = true; $('login-error').hidden = true;
    const username = $('login-username').value.trim(), password = $('login-password').value;
    try {
      const payload = await fetchJson('/api/login', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }) }, true);
      if (!validSession(payload)) throw new Error('Не удалось подтвердить вход.');
      state.manualLogin = false;
      await acceptSession(payload, true);
      if (state.user) $('question').focus();
    } catch (error) {
      signedOut(error.message || 'Не удалось войти.');
    } finally {
      $('login-password').value = ''; $('login-submit').disabled = false; state.authBusy = false;
    }
  }
  async function logout() {
    if (state.authBusy || !state.user) return;
    state.authBusy = true; state.manualLogin = true; state.sessionRequestId += 1;
    clearWorkspace(); $('workspace').hidden = true; $('logout-button').disabled = true;
    try {
      await fetchJson('/api/logout', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }, true);
      signedOut(); $('login-username').focus();
    } catch (_) {
      signedOut('Не удалось подтвердить выход на сервере. Войдите снова, когда соединение восстановится.');
    } finally { state.authBusy = false; $('logout-button').disabled = false; }
  }
  function accessLabel(value) {
    return value === 2 ? 'Уровень 2' : value === 1 ? 'Уровень 1' : 'Без пометки · доступен уровню 1';
  }
  function renderAccess(document) {
    $('document-access').hidden = false;
    $('document-access-label').textContent = accessLabel(document.access_level);
    $('document-access-controls').hidden = state.user?.level !== 2;
    $('document-access-level').value = document.access_level === 2 ? '2' : '1';
    $('document-access-save').disabled = false; $('document-access-message').hidden = true;
  }
  async function saveDocumentAccess() {
    if (state.user?.level !== 2 || !state.currentDocument) return;
    const id = state.currentDocument.id, epoch = state.authEpoch;
    $('document-access-save').disabled = true; $('document-access-message').hidden = true;
    try {
      await fetchJson(`/api/documents/${encodeURIComponent(id)}/access`, { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ access_level: Number($('document-access-level').value) }) });
      if (epoch !== state.authEpoch) return;
      await refreshSession(true);
      if (state.user?.level === 2) { await openDocument(id); toast('Уровень доступа сохранён. Контекст диалогов обновлён.'); }
    } catch (error) {
      if (epoch !== state.authEpoch) return;
      $('document-access-message').textContent = error.message;
      $('document-access-message').className = 'access-error'; $('document-access-message').hidden = false;
    } finally { if (epoch === state.authEpoch) $('document-access-save').disabled = false; }
  }

  function updateMode(health) {
    state.mode = health.mode === 'llm' ? 'llm' : 'extractive';
    $('mode-label').textContent = state.mode === 'llm' ? 'API настроен' : 'Поиск по документам';
    $('mode-status').title = state.mode === 'llm' && health.model ? `Модель: ${health.model}` : 'Поиск и цитирование положений учебной базы';
    $('mode-status').querySelector('.status-dot').className = 'status-dot';
    $('composer-footnote').textContent = state.mode === 'extractive'
      ? 'Поиск по документам. Для свободных ответов подключите модель.'
      : 'Учебный корпус. Проверяйте применимость положений по источникам.';
    $('connection-notice').hidden = true;
  }
  function updateDocumentCount(count) {
    $('sidebar-document-count').textContent = String(count);
    $('sidebar-document-count').nextElementSibling.textContent = `${plural(count, ['документ', 'документа', 'документов'])} в базе`;
    $('source-count').textContent = String(count);
    $('source-count').title = `${count} ${plural(count, ['документ', 'документа', 'документов'])} в базе`;
  }
  async function loadHealth() {
    const epoch = state.authEpoch;
    if (!state.user) return;
    try { const health = await fetchJson('/api/health'); if (epoch === state.authEpoch && state.user) updateMode(health); }
    catch (_) {
      if (epoch !== state.authEpoch || !state.user) return;
      $('mode-label').textContent = 'Нет подключения';
      $('mode-status').querySelector('.status-dot').className = 'status-dot error';
      $('connection-message').textContent = 'Не удалось подключиться к базе. Проверьте, что сервер запущен.';
      $('connection-notice').hidden = false;
    }
  }
  async function loadDocuments() {
    const epoch = state.authEpoch;
    if (!state.user) return;
    try {
      const data = await fetchJson('/api/documents');
      if (epoch !== state.authEpoch || !state.user) return;
      if (Number.isInteger(data.acl_revision) && data.acl_revision !== state.aclRevision) { await refreshSession(true); return; }
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
      if (epoch !== state.authEpoch || !state.user) return;
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
      copy.append(element('strong', '', titleOf(doc)), element('small', '', kindLabel(doc.kind)), element('span', 'access-badge', accessLabel(doc.access_level)));
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
    if (!state.user) return;
    const request = ++state.documentRequestId;
    state.currentDocument = null;
    $('document-access').hidden = true;
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
      // Reauthorize every document open; a cached text is not an access grant.
      const doc = await fetchJson(`/api/documents/${encodeURIComponent(id)}`);
      if (request !== state.documentRequestId || !state.user) return;
      state.currentDocument = doc;
      $('document-view-title').textContent = titleOf(doc);
      $('copy-document').disabled = false;
      renderAccess(doc);
      renderDocumentText(doc, citation);
    } catch (error) {
      if (request !== state.documentRequestId) return;
      $('document-text').classList.add('document-error');
      $('document-text').textContent = 'Не удалось открыть документ. Вернитесь к списку и попробуйте ещё раз.';
      $('document-view-description').textContent = error.status === 404 ? 'Документ недоступен для вашей учётной записи' : 'Источник временно недоступен';
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
    if (!state.user || !state.historyLoaded || state.pending || !question.trim()) return;
    const epoch = state.authEpoch;
    await refreshSession();
    if (epoch !== state.authEpoch || !state.user || state.pending) return;
    question = question.trim();
    const chat = conversations.active();
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
        body: JSON.stringify({ question, event_date: date || null, conversation_id: chat.id, turn_id: turn.id, acl_revision: state.aclRevision }), signal: controller.signal,
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
    if (!state.user || !state.historyLoaded) return;
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
      remove.addEventListener('click', async () => {
        const epoch = state.authEpoch;
        if (chat.id === conversations.data.activeId) cancelPending();
        remove.disabled = true;
        try {
          await fetchJson('/api/history/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conversation_id: chat.id }) });
          if (epoch !== state.authEpoch) return;
          conversations.remove(chat.id); renderHistory(); renderConversation(); toast('Диалог удалён');
        } catch (error) { if (epoch === state.authEpoch) toast(error.message); }
        finally { if (epoch === state.authEpoch) remove.disabled = false; }
      });
      row.append(open, remove); list.append(row);
    }
  }
  function openHistory() {
    if (!state.user || !state.historyLoaded) return;
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
  $('confirm-clear-history').addEventListener('click', async () => {
    const epoch = state.authEpoch;
    cancelPending(); $('confirm-clear-history').disabled = true;
    try {
      await fetchJson('/api/history/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      if (epoch !== state.authEpoch) return;
      conversations.clear(); renderHistory(); renderConversation();
      $('history-delete-confirm').hidden = true; toast('История вашей учётной записи удалена');
    } catch (error) { if (epoch === state.authEpoch) toast(error.message); }
    finally { if (epoch === state.authEpoch) $('confirm-clear-history').disabled = false; }
  });
  $('retry-storage').addEventListener('click', () => { if (conversations.save()) toast('История сохранена'); });
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
  $('retry-connection').addEventListener('click', () => { refreshSession(true); });
  window.addEventListener('resize', () => {
    if (!window.matchMedia('(max-width: 760px)').matches) toggleSourcePanel(false);
    else if ($('source-panel').classList.contains('is-open')) $('source-backdrop').hidden = false;
  });
  $('login-form').addEventListener('submit', login);
  $('logout-button').addEventListener('click', logout);
  $('document-access-save').addEventListener('click', saveDocumentAccess);
  window.addEventListener('focus', () => { refreshSession(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refreshSession(); });
  setInterval(() => { if (!document.hidden) refreshSession(); }, 15000);
  $('mobile-sources').hidden = true;
  refreshSession();
})();
