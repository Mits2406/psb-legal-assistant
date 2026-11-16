'use strict';

// Persist only displayable chat data, never credentials or raw API diagnostics.
// This small module is also exercised directly by the Node regression tests.
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.PSBHistory = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const KEY = 'psb.conversations.v1';
  const MAX_RAW_CHARS = 5000000;
  const interrupted = 'Запрос прервался. Вопрос сохранён — его можно повторить.';
  const isObject = value => Boolean(value && typeof value === 'object' && !Array.isArray(value));
  const text = (value, max = 12000) => typeof value === 'string' ? value.slice(0, max) : '';
  const id = () => typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID() : Date.now().toString(36) + '-' + Math.random().toString(36).slice(2);

  function cleanDialogueState(value) {
    if (!isObject(value) || value.version !== 1) return null;
    // State is data, never executable or trusted evidence. The backend checks
    // its user-message provenance again before using an outstanding request.
    try {
      const raw = JSON.stringify(value);
      return raw.length <= 40000 ? JSON.parse(raw) : null;
    } catch (_) { return null; }
  }

  function cleanAnswer(value) {
    if (!isObject(value) || typeof value.answer !== 'string' || !value.answer.trim()) return null;
    const answer = {
      answer: text(value.answer),
      status: value.status === 'technical_error' ? 'error' : ['answer', 'partial', 'clarify', 'abstain', 'out_of_scope', 'error'].includes(value.status) ? value.status : 'answer',
      mode: value.mode === 'llm' ? 'llm' : 'extractive',
      citations: [], warnings: [], missing_facts: [],
    };
    for (const item of (Array.isArray(value.citations) ? value.citations : []).slice(0, 100)) {
      if (!isObject(item) || typeof item.doc_id !== 'string' || !/^doc_[a-f0-9]{16}$/.test(item.doc_id)) continue;
      const citation = {};
      for (const key of ['id', 'source_id', 'doc_id', 'title', 'clause', 'source_clause', 'display_label', 'quote']) citation[key] = text(item[key], key === 'quote' ? 12000 : 1500);
      for (const key of ['start', 'end']) if (Number.isSafeInteger(item[key]) && item[key] >= 0) citation[key] = item[key];
      answer.citations.push(citation);
    }
    for (const key of ['warnings', 'missing_facts']) {
      answer[key] = (Array.isArray(value[key]) ? value[key] : []).filter(item => typeof item === 'string').slice(0, 30).map(item => text(item, 2000));
    }
    if (isObject(value.self_check)) answer.self_check = {
      enabled: value.self_check.enabled === true,
      status: text(value.self_check.status, 80),
      completed: value.self_check.completed === true,
      changed: value.self_check.changed === true,
    };
    const dialogueState = cleanDialogueState(value.dialogue_state);
    if (dialogueState) answer.dialogue_state = dialogueState;
    if (isObject(value.trace)) {
      answer.trace = {};
      if (Number.isFinite(value.trace.total_ms) && value.trace.total_ms >= 0) answer.trace.total_ms = value.trace.total_ms;
      const generation = value.trace.generation;
      if (isObject(generation)) {
        answer.trace.generation = {};
        for (const key of ['semantic_audited', 'self_check_passed']) {
          if (typeof generation[key] === 'boolean') answer.trace.generation[key] = generation[key];
        }
        if (typeof generation.self_check_status === 'string') answer.trace.generation.self_check_status = text(generation.self_check_status, 80);
      }
    }
    return answer;
  }

  function cleanTurn(value) {
    if (!isObject(value) || typeof value.id !== 'string' || typeof value.question !== 'string' || !value.question.trim()) return null;
    const result = {
      id: text(value.id, 100), question: text(value.question),
      date: typeof value.date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value.date) ? value.date : null,
      at: Number.isFinite(value.at) ? value.at : 0,
      status: ['pending', 'complete', 'failed', 'interrupted'].includes(value.status) ? value.status : 'interrupted',
    };
    if (result.status === 'complete') {
      result.answerData = cleanAnswer(value.answerData);
      if (!result.answerData) result.status = 'interrupted';
    }
    if (result.status === 'failed' || result.status === 'interrupted') result.error = text(value.error, 2000) || interrupted;
    return result;
  }

  function decode(raw) {
    if (!raw) return { version: 1, activeId: null, chats: [] };
    if (raw.length > MAX_RAW_CHARS) throw new Error('storage_too_large');
    const value = JSON.parse(raw);
    if (!isObject(value) || value.version !== 1 || !Array.isArray(value.chats)) throw new Error('storage_shape');
    const seen = new Set();
    const chats = value.chats.filter(chat => isObject(chat) && typeof chat.id === 'string' && Array.isArray(chat.turns)).map(chat => ({
      id: text(chat.id, 100),
      createdAt: Number.isFinite(chat.createdAt) ? chat.createdAt : 0,
      updatedAt: Number.isFinite(chat.updatedAt) ? chat.updatedAt : 0,
      turns: chat.turns.map(cleanTurn).filter(Boolean),
    })).filter(chat => {
      if (!chat.id || seen.has(chat.id)) return false;
      seen.add(chat.id);
      const turnIds = new Set();
      chat.turns = chat.turns.filter(turn => !turnIds.has(turn.id) && Boolean(turnIds.add(turn.id)));
      return true;
    });
    return { version: 1, activeId: chats.some(chat => chat.id === value.activeId) ? value.activeId : chats[0]?.id || null, chats };
  }

  function context(chat, beforeTurnId = null) {
    const messages = [];
    for (const turn of chat?.turns || []) {
      if (turn.id === beforeTurnId) break;
      if (turn.status !== 'complete' || !turn.answerData) continue;
      messages.push({ role: 'user', content: turn.question + (turn.date ? `\nДата события: ${turn.date}` : '') });
      // A transport error or a diagnostic marker is not a model answer or a fact.
      if (turn.answerData.status !== 'error' && turn.answerData.status !== 'technical_error' && !/^DIAG_/.test(turn.answerData.answer)) {
        messages.push({ role: 'assistant', content: turn.answerData.answer });
      }
    }
    let result = messages.slice(-16);
    if (result[0]?.role === 'assistant') result.shift();
    while (result.length && result.reduce((sum, item) => sum + item.content.length, 0) > 60000) {
      result.shift();
      if (result[0]?.role === 'assistant') result.shift();
    }
    return result;
  }

  function dialogueState(chat, beforeTurnId = null) {
    let state = null;
    for (const turn of chat?.turns || []) {
      if (turn.id === beforeTurnId) break;
      if (turn.status !== 'complete' || !turn.answerData) continue;
      const saved = cleanDialogueState(turn.answerData.dialogue_state);
      if (saved) state = saved;
    }
    return state;
  }

  // Three-way merging retains changes made by another tab to different chats
  // and different turns. Deletion wins over a late response in a removed chat.
  function merge(local, remote, base) {
    const baseChats = new Map(base.chats.map(chat => [chat.id, chat]));
    const localChats = new Map(local.chats.map(chat => [chat.id, chat]));
    const chats = new Map(remote.chats.map(chat => [chat.id, chat]));
    const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
    for (const [chatId, oldChat] of baseChats) if (!localChats.has(chatId)) chats.delete(chatId);
    for (const [chatId, localChat] of localChats) {
      const oldChat = baseChats.get(chatId), remoteChat = chats.get(chatId);
      if (!oldChat) { chats.set(chatId, localChat); continue; }
      if (!remoteChat || same(localChat, oldChat)) continue;
      const oldTurns = new Map(oldChat.turns.map(turn => [turn.id, turn]));
      const turns = new Map(remoteChat.turns.map(turn => [turn.id, turn]));
      for (const turn of localChat.turns) {
        if (!oldTurns.has(turn.id) || !same(turn, oldTurns.get(turn.id))) turns.set(turn.id, turn);
      }
      chats.set(chatId, { ...localChat, updatedAt: Math.max(localChat.updatedAt, remoteChat.updatedAt),
        turns: [...turns.values()].sort((a, b) => a.at - b.at) });
    }
    const result = { version: 1, activeId: local.activeId, chats: [...chats.values()] };
    if (!chats.has(result.activeId)) result.activeId = chats.has(remote.activeId) ? remote.activeId : result.chats[0]?.id || null;
    return result;
  }

  class Store {
    constructor(storage, onWarning = () => {}, key = KEY) {
      this.storage = storage;
      this.key = key;
      this.onWarning = onWarning;
      this.data = { version: 1, activeId: null, chats: [] };
      this.readFailed = false;
      try { this.data = decode(storage.getItem(this.key)); }
      catch (_) {
        this.readFailed = true;
        onWarning('Не удалось прочитать сохранённую историю. Она оставлена в браузере; новые сообщения пока хранятся только в этой вкладке.');
      }
      this.base = JSON.parse(JSON.stringify(this.data));
      // Pending requests cannot survive a navigation. Keep their question and
      // offer an explicit retry instead of pretending a response was received.
      let changed = false;
      for (const chat of this.data.chats) for (const turn of chat.turns) if (turn.status === 'pending') {
        turn.status = 'interrupted'; turn.error = interrupted; changed = true;
      }
      if (!this.active()) this.newChat();
      else if (changed) this.save();
    }

    active() { return this.data.chats.find(chat => chat.id === this.data.activeId) || null; }
    chat(chatId) { return this.data.chats.find(chat => chat.id === chatId) || null; }
    turn(chatId, turnId) { return this.chat(chatId)?.turns.find(turn => turn.id === turnId) || null; }

    refresh() {
      if (this.readFailed) return;
      try {
        const remote = decode(this.storage.getItem(this.key));
        this.data = merge(this.data, remote, this.base);
        this.base = JSON.parse(JSON.stringify(remote));
      } catch (_) { /* Keep the last readable history; save() reports failures. */ }
    }

    save() {
      if (this.readFailed) return false;
      try {
        const remote = decode(this.storage.getItem(this.key));
        this.data = merge(this.data, remote, this.base);
        const raw = JSON.stringify(this.data);
        if (raw.length > MAX_RAW_CHARS) throw new Error('storage_too_large');
        this.storage.setItem(this.key, raw);
        this.base = JSON.parse(raw);
        this.onWarning('');
        return true;
      } catch (_) {
        this.onWarning('Не удалось сохранить новые сообщения в браузере: хранилище недоступно или заполнено. Не закрывайте вкладку; удалите ненужные диалоги и повторите сохранение.');
        return false;
      }
    }

    newChat() {
      const active = this.active();
      if (active && !active.turns.length) return active;
      const chat = { id: id(), createdAt: Date.now(), updatedAt: Date.now(), turns: [] };
      this.data.chats.push(chat); this.data.activeId = chat.id; this.save();
      return chat;
    }

    select(chatId) {
      if (!this.chat(chatId)) return false;
      this.data.activeId = chatId; this.save(); return true;
    }

    begin(question, date = null, retryId = null) {
      const chat = this.active();
      if (!chat) throw new Error('No active conversation');
      let turn = retryId ? this.turn(chat.id, retryId) : null;
      if (retryId && !turn) throw new Error('Missing retry turn');
      if (!turn) {
        turn = { id: id(), question: text(question), date, at: Date.now(), status: 'pending' };
        chat.turns.push(turn);
      } else {
        turn.status = 'pending'; delete turn.error; delete turn.answerData;
      }
      chat.updatedAt = Date.now(); this.save();
      return turn;
    }

    complete(chatId, turnId, value) {
      const turn = this.turn(chatId, turnId);
      const answer = cleanAnswer(value);
      if (!turn || !answer) return false;
      turn.status = 'complete'; turn.answerData = answer; delete turn.error;
      this.chat(chatId).updatedAt = Date.now(); this.save(); return true;
    }

    fail(chatId, turnId, error, status = 'failed') {
      const turn = this.turn(chatId, turnId);
      if (!turn) return false;
      turn.status = status === 'interrupted' ? 'interrupted' : 'failed';
      turn.error = text(error, 2000) || interrupted; delete turn.answerData;
      this.chat(chatId).updatedAt = Date.now(); this.save(); return true;
    }

    remove(chatId) {
      this.data.chats = this.data.chats.filter(chat => chat.id !== chatId);
      if (this.data.activeId === chatId) this.data.activeId = this.data.chats.slice().sort((a, b) => b.updatedAt - a.updatedAt)[0]?.id || null;
      if (!this.active()) this.newChat();
      else this.save();
    }

    clear() {
      // User initiated deletion is also the recovery path for an invalid store.
      this.readFailed = false;
      try { this.storage.removeItem(this.key); } catch (_) { /* save reports the failure */ }
      this.base = JSON.parse(JSON.stringify(this.data));
      this.data = { version: 1, activeId: null, chats: [] }; this.newChat();
    }
  }

  return { KEY, Store, context, dialogueState, decode, cleanAnswer, interrupted, merge };
});
