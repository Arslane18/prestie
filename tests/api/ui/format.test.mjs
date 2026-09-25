// Unit tests for the UI's pure helpers: `node --test tests/api/ui/format.test.mjs`
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  ageMinutes,
  classColor,
  createSseParser,
  escapeHtml,
  renderMarkdown,
  stepLabel,
} from "../../../src/prestie/api/static/format.js";

test("escapeHtml neutralizes markup", () => {
  assert.equal(
    escapeHtml(`<img src=x onerror="alert(1)">&'`),
    "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;&amp;&#39;",
  );
});

test("renderMarkdown never lets HTML through", () => {
  const html = renderMarkdown("<script>alert(1)</script> **<b>gras</b>**");
  assert.ok(!html.includes("<script>"));
  assert.ok(!html.includes("<b>"));
  assert.ok(html.includes("<strong>&lt;b&gt;gras&lt;/b&gt;</strong>"));
});

test("renderMarkdown renders paragraphs, bold and bullet lists", () => {
  const html = renderMarkdown("Intro **Haste**\n\n- un\n- deux\nFin");
  assert.equal(
    html,
    "<p>Intro <strong>Haste</strong></p><ul><li>un</li><li>deux</li></ul><p>Fin</p>",
  );
});

test("renderMarkdown links source URLs and leaves trailing punctuation out", () => {
  const html = renderMarkdown("- Stats — https://www.icy-veins.com/wow/a#b.");
  assert.ok(
    html.includes(
      '<a href="https://www.icy-veins.com/wow/a#b" target="_blank" rel="noopener noreferrer">https://www.icy-veins.com/wow/a#b</a>.',
    ),
  );
});

test("renderMarkdown does not link javascript: or quoted URLs", () => {
  const html = renderMarkdown(`javascript:alert(1) "https://x.y/"onmouseover="z"`);
  assert.ok(!html.includes('href="javascript'));
  assert.ok(!html.includes('onmouseover="'));
});

test("SSE parser handles chunks split anywhere, CRLF and comments", () => {
  const events = [];
  const feed = createSseParser((event) => events.push(event));
  feed('event: text\r\ndata: {"text": "Ha');
  feed('ste"}\r\n\r\n: ping\r\n\r\nevent: done\ndata: {"text": "ok"}\n\n');
  assert.deepEqual(events, [
    { event: "text", data: { text: "Haste" } },
    { event: "done", data: { text: "ok" } },
  ]);
});

test("SSE parser skips malformed data", () => {
  const events = [];
  const feed = createSseParser((event) => events.push(event));
  feed("event: text\ndata: {not json\n\n");
  assert.deepEqual(events, []);
});

test("ageMinutes counts whole minutes and never goes negative", () => {
  const now = Date.parse("2026-09-25T20:00:00Z");
  assert.equal(ageMinutes("2026-09-25T19:47:30+00:00", now), 12);
  assert.equal(ageMinutes("2026-09-25T20:05:00+00:00", now), 0);
});

test("classColor uses the official class colors and a gold fallback", () => {
  assert.equal(classColor("DEATHKNIGHT"), "#C41E3A");
  assert.equal(classColor("PRIEST"), "#FFFFFF");
  assert.equal(classColor("UNKNOWN"), null);
  assert.equal(classColor(undefined), null);
});

test("stepLabel turns CLI tags into readable labels", () => {
  assert.equal(stepLabel("[recherche] stat priority"), "Recherche\u00a0: stat priority");
  assert.equal(stepLabel("[quête] détails de la quête 55881"), "Quête\u00a0: détails de la quête 55881");
  assert.equal(stepLabel("je vérifie d'abord"), "Je vérifie d'abord");
});
