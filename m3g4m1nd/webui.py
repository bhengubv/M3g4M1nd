"""A minimal chat page, served from the same process as the API.

MiniMind's web demo is a Streamlit app — a second dependency, a second server
and a second copy of the streaming parser. This is one HTML string served at
`/` by the same FastAPI app, talking to the same `/v1/chat/completions`
endpoint over server-sent events. No new dependency, nothing to keep in sync,
and if the stream parser is wrong the API tests catch it rather than a human
noticing the page looks odd.

It is deliberately plain. The point is that the thing can be used, not that it
is pretty.
"""

from __future__ import annotations

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>M3g4M1nd</title>
<style>
  :root {
    --bg: #faf9f7; --fg: #1a1f36; --muted: #6b7280;
    --line: #e5e3df; --accent: #2563eb; --panel: #ffffff;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14161a; --fg: #e8e6e3; --muted: #9aa0aa;
      --line: #2a2d34; --accent: #6b9dff; --panel: #1b1e24;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  header {
    padding: 14px 16px; border-bottom: 1px solid var(--line);
    display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
  }
  h1 { font-size: 16px; margin: 0; letter-spacing: -0.01em; }
  header span { color: var(--muted); font-size: 13px; }
  main { max-width: 760px; margin: 0 auto; padding: 16px; }
  #log { display: flex; flex-direction: column; gap: 12px; min-height: 40vh; }
  .turn { padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px;
          background: var(--panel); white-space: pre-wrap; word-wrap: break-word; }
  .turn.user { background: transparent; border-style: dashed; }
  .who { font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
         color: var(--muted); margin-bottom: 4px; }
  .reasoning { color: var(--muted); font-style: italic; }
  .tool { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px;
          color: var(--accent); }
  form { display: flex; gap: 8px; margin-top: 16px; }
  input[type=text] {
    flex: 1; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px;
    background: var(--panel); color: var(--fg); font: inherit; min-width: 0;
  }
  button { padding: 10px 16px; border: 0; border-radius: 8px; background: var(--accent);
           color: #fff; font: inherit; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  label { display: flex; align-items: center; gap: 6px; color: var(--muted);
          font-size: 13px; margin-top: 10px; }
</style>
</head>
<body>
<header>
  <h1>M3g4M1nd</h1>
  <span id="status">checking&hellip;</span>
</header>
<main>
  <div id="log"></div>
  <form id="ask">
    <input id="text" type="text" placeholder="Ask something" autocomplete="off" autofocus>
    <button id="send" type="submit">Send</button>
  </form>
  <label><input id="thinking" type="checkbox"> show reasoning</label>
</main>
<script>
const log = document.getElementById('log');
const form = document.getElementById('ask');
const field = document.getElementById('text');
const send = document.getElementById('send');
const thinking = document.getElementById('thinking');
const status = document.getElementById('status');
const history = [];

fetch('health').then(r => r.json()).then(d => {
  status.textContent = d.model_loaded ? 'model loaded' : 'no model loaded';
}).catch(() => { status.textContent = 'unreachable'; });

function turn(who, cssClass) {
  const box = document.createElement('div');
  box.className = 'turn ' + (cssClass || '');
  const label = document.createElement('div');
  label.className = 'who';
  label.textContent = who;
  const body = document.createElement('div');
  box.append(label, body);
  log.append(box);
  box.scrollIntoView({block: 'end'});
  return body;
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  const question = field.value.trim();
  if (!question) return;
  field.value = '';
  send.disabled = true;
  turn('you', 'user').textContent = question;
  history.push({role: 'user', content: question});

  const body = turn('m3g4m1nd');
  const reasoning = document.createElement('div');
  reasoning.className = 'reasoning';
  const answer = document.createElement('div');
  body.append(reasoning, answer);
  let collected = '';

  try {
    const response = await fetch('v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        model: 'm3g4m1nd', messages: history, stream: true,
        open_thinking: thinking.checked, max_tokens: 256
      })
    });
    if (!response.ok) throw new Error('HTTP ' + response.status);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (!payload || payload === '[DONE]') continue;
        let frame;
        try { frame = JSON.parse(payload); } catch { continue; }
        if (frame.error) { answer.textContent += '\\n[error] ' + (frame.error.message || ''); continue; }
        const delta = (frame.choices && frame.choices[0] && frame.choices[0].delta) || {};
        if (delta.reasoning_content) reasoning.textContent += delta.reasoning_content;
        if (delta.content) { answer.textContent += delta.content; collected += delta.content; }
        for (const call of delta.tool_calls || []) {
          const line = document.createElement('div');
          line.className = 'tool';
          line.textContent = '-> ' + call.function.name + '(' + call.function.arguments + ')';
          body.append(line);
        }
      }
      body.scrollIntoView({block: 'end'});
    }
    history.push({role: 'assistant', content: collected});
  } catch (error) {
    answer.textContent += '\\n[failed] ' + error.message;
  } finally {
    send.disabled = false;
    field.focus();
  }
});
</script>
</body>
</html>
"""
