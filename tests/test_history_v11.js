'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { Store, KEY, context, dialogueState, cleanAnswer } = require('../static/history.js');
function storage() {
  const values = new Map();
  return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key), values };
}
const source = { source_id: 'S003', id: 'chunk_12', doc_id: 'doc_0123456789abcdef', title: 'Договор', clause: '2.1', quote: 'Полный фрагмент', start: 20, end: 34 };
const response = { answer: 'Договор применяется с указанной даты. [S003]', status: 'answer', mode: 'llm', citations: [source], self_check: { enabled: true, status: 'completed', completed: true, changed: false }, trace: { total_ms: 1234, generation: { api_key: 'DO_NOT_SAVE', semantic_audited: true } } };
function complete(store, question = 'Что в договоре?', date = null, answer = response) {
  const chatId = store.active().id;
  const turn = store.begin(question, date);
  store.complete(chatId, turn.id, answer);
  return turn.id;
}

test('pending interpretation survives reload and stays inside its conversation', () => {
  const disk = storage(), store = new Store(disk);
  const saved = { version: 1, original_task: 'Кто согласует договор?', pending_fields: ['event_date'], history_digest: 'test' };
  complete(store, saved.original_task, null, { ...response, dialogue_state: saved });
  const firstId = store.active().id;
  const restored = new Store(disk);
  assert.deepEqual(dialogueState(restored.active()), saved);
  restored.newChat();
  assert.equal(dialogueState(restored.active()), null);
  restored.select(firstId);
  const reply = restored.begin('Сейчас.');
  assert.deepEqual(dialogueState(restored.active(), reply.id), saved);
  restored.complete(firstId, reply.id, { ...response, dialogue_state: { ...saved, pending_fields: [] } });
  assert.deepEqual(dialogueState(restored.active()).pending_fields, []);
  assert.deepEqual(dialogueState(restored.active(), reply.id).pending_fields, ['event_date']);
});

test('invalid or oversized interpretation state cannot expand the request body', () => {
  assert.equal(cleanAnswer({ ...response, dialogue_state: { version: 2 } }).dialogue_state, undefined);
  assert.equal(cleanAnswer({ ...response, dialogue_state: { version: 1, text: 'x'.repeat(40001) } }).dialogue_state, undefined);
});

test('persists question before response, reload recovers interrupted turn and retry uses same ID', () => {
  const disk = storage(), store = new Store(disk);
  const turn = store.begin('Как согласовать договор?', '2025-02-01');
  const serialized = JSON.parse(disk.getItem(KEY));
  assert.equal(serialized.chats[0].turns[0].question, 'Как согласовать договор?');
  assert.equal(serialized.chats[0].turns[0].status, 'pending');
  const reload = new Store(disk);
  assert.equal(reload.active().turns[0].status, 'interrupted');
  assert.deepEqual(context(reload.active()), []);
  const retry = reload.begin(turn.question, turn.date, turn.id);
  assert.equal(retry.id, turn.id);
  assert.equal(reload.active().turns.length, 1);
  reload.complete(reload.active().id, turn.id, response);
  assert.equal(new Store(disk).active().turns.length, 1);
  assert.equal(new Store(disk).active().turns[0].status, 'complete');
});

test('restores full answer, citation identity/offsets, event date, and actual self-check result', () => {
  const disk = storage(), store = new Store(disk);
  complete(store, 'Какой адрес?', '2024-10-02');
  const restored = new Store(disk).active().turns[0];
  assert.equal(restored.date, '2024-10-02');
  assert.equal(restored.answerData.answer, response.answer);
  assert.deepEqual(restored.answerData.citations[0], { ...source, source_clause: '', display_label: '' });
  assert.equal(restored.answerData.self_check.completed, true);
  assert.equal(restored.answerData.trace.total_ms, 1234);
  assert.equal(disk.getItem(KEY).includes('DO_NOT_SAVE'), false);
});

test('new chat archives preceding one, restores selected chat, isolates model context and deletes history', () => {
  const disk = storage(), store = new Store(disk);
  complete(store, 'Договор А-08/2024');
  const firstId = store.active().id;
  store.newChat();
  assert.deepEqual(context(store.active()), []);
  complete(store, 'Другой договор');
  const secondId = store.active().id;
  assert.notEqual(firstId, secondId);
  assert.equal(store.data.chats.length, 2);
  store.select(firstId);
  const reload = new Store(disk);
  assert.equal(reload.active().id, firstId);
  assert.equal(context(reload.active())[0].content, 'Договор А-08/2024');
  reload.remove(firstId);
  assert.equal(new Store(disk).data.chats.some(chat => chat.id === firstId), false);
  reload.clear();
  assert.equal(new Store(disk).data.chats.length, 1);
  assert.deepEqual(new Store(disk).active().turns, []);
});

test('failed requests and technical diagnostic answers do not enter model context as assistant facts', () => {
  const store = new Store(storage());
  const failed = store.begin('Первая попытка');
  store.fail(store.active().id, failed.id, 'Сеть недоступна');
  complete(store, 'Вторая попытка', null, { answer: 'DIAG_API_FAILURE: timeout', status: 'technical_error', mode: 'diagnostic' });
  complete(store, 'Уточнение', '2025-02-01', { answer: 'Уточните номер договора.', status: 'clarify', mode: 'llm' });
  const history = context(store.active());
  assert.equal(history.some(message => message.content.includes('Первая попытка')), false);
  assert.equal(history.some(message => message.content.includes('DIAG_')), false);
  assert.equal(history.some(message => message.content.includes('Дата события: 2025-02-01')), true);
  assert.equal(history.at(-1).content, 'Уточните номер договора.');
});

test('retry of earlier question excludes future turns and cannot duplicate the user turn', () => {
  const store = new Store(storage());
  complete(store, 'Вопрос 1');
  const failed = store.begin('Вопрос 2');
  store.fail(store.active().id, failed.id, 'Ошибка');
  complete(store, 'Вопрос 3');
  assert.equal(context(store.active(), failed.id).length, 2);
  store.begin('Вопрос 2', null, failed.id);
  assert.equal(store.active().turns.length, 3);
  assert.equal(store.active().turns.filter(turn => turn.question === 'Вопрос 2').length, 1);
});

test('context obeys count and total character budget, retains complete adjacent pairs', () => {
  const store = new Store(storage());
  for (let index = 0; index < 12; index++) complete(store, `Вопрос ${index}: ` + 'а'.repeat(11000), null, { answer: `Ответ ${index}: ` + 'б'.repeat(9000), status: 'answer', mode: 'llm' });
  const history = context(store.active());
  assert.ok(history.length <= 16);
  assert.ok(history.reduce((sum, message) => sum + message.content.length, 0) <= 60000);
  assert.equal(history[0].role, 'user');
  assert.equal(history.at(-1).role, 'assistant');
  assert.ok(history.at(-1).content.startsWith('Ответ 11:'));
  assert.equal(store.active().turns.length, 12); // Display history is not truncated to model budget.
});

test('corrupt storage is not overwritten until user explicitly clears it', () => {
  const disk = storage(), warnings = [];
  disk.setItem(KEY, '{corrupt');
  const store = new Store(disk, warning => warnings.push(warning));
  complete(store);
  assert.equal(disk.getItem(KEY), '{corrupt');
  assert.ok(warnings.some(Boolean));
  assert.equal(store.active().turns.length, 1);
  store.clear();
  assert.doesNotThrow(() => JSON.parse(disk.getItem(KEY)));
  assert.equal(store.readFailed, false);
});

test('quota failure keeps unsaved conversation in memory and older data on disk; save can be retried', () => {
  const disk = storage(), warnings = [], store = new Store(disk, value => warnings.push(value));
  complete(store, 'Сохранённый вопрос');
  const snapshot = disk.getItem(KEY), write = disk.setItem;
  disk.setItem = () => { throw new Error('QuotaExceededError'); };
  complete(store, 'Новый вопрос');
  assert.equal(store.active().turns.length, 2);
  assert.equal(disk.getItem(KEY), snapshot);
  assert.ok(warnings.at(-1));
  disk.setItem = write;
  assert.equal(store.save(), true);
  assert.equal(new Store(disk).active().turns.length, 2);
  assert.equal(warnings.at(-1), '');
});

test('another tab creating a chat cannot erase a response to an older in-flight turn', () => {
  const disk = storage(), first = new Store(disk);
  const pending = first.begin('Договор П-17/2024');
  const firstId = first.active().id;
  const second = new Store(disk);
  second.newChat(); complete(second, 'Вопрос из другой вкладки');
  const secondId = second.active().id;
  first.complete(firstId, pending.id, response);
  const restore = new Store(disk);
  assert.equal(restore.data.chats.length, 2);
  assert.equal(restore.chat(firstId).turns[0].status, 'complete');
  assert.equal(restore.chat(secondId).turns[0].question, 'Вопрос из другой вкладки');
  // A stale tab saving selection must keep the new response, too.
  second.select(secondId);
  assert.equal(new Store(disk).chat(firstId).turns[0].status, 'complete');
});

test('deletion in another tab wins over a late response and does not resurrect private messages', () => {
  const disk = storage(), first = new Store(disk);
  const turn = first.begin('Удаляемый вопрос'), firstId = first.active().id;
  const second = new Store(disk);
  second.remove(firstId);
  first.complete(firstId, turn.id, response);
  assert.equal(new Store(disk).chat(firstId), null);
});

test('invalid stored citations and credential fields are discarded, user text stays plain data', () => {
  const result = cleanAnswer({ ...response, citations: [source, { doc_id: 'javascript:alert(1)', quote: '<script>bad()</script>' }], api_key: 'secret', answer: '<script>text only</script>' });
  assert.equal(result.citations.length, 1);
  assert.equal(result.answer, '<script>text only</script>');
  assert.equal(Object.hasOwn(result, 'api_key'), false);
  assert.equal(cleanAnswer({ answer: '' }), null);
});
