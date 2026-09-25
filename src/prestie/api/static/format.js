// Pure helpers for the companion window (no DOM): tested with `node --test`.

const HTML_ESCAPES = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};
const BULLET = /^\s*[-*•]\s+/;
const BOLD = /\*\*(.+?)\*\*/g;
// Runs on escaped text: a URL stops at whitespace, "<" or an escaped quote,
// and trailing punctuation is left outside the link.
const URL = /https?:\/\/[^\s<]+?(?=[.,;:!?)]*(?:\s|$|<|&quot;|&#39;))/g;
const MS_PER_MINUTE = 60_000;

export function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (char) => HTML_ESCAPES[char]);
}

function inline(escaped) {
  return escaped
    .replace(BOLD, "<strong>$1</strong>")
    .replace(
      URL,
      (url) =>
        `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`,
    );
}

/** Minimal Markdown for answers: paragraphs, **bold**, "- " lists, links.
 * Everything is HTML-escaped first, so model or game text cannot inject markup. */
export function renderMarkdown(text) {
  const parts = [];
  let inList = false;
  for (const line of escapeHtml(text).split("\n")) {
    const isItem = BULLET.test(line);
    if (isItem && !inList) parts.push("<ul>");
    if (!isItem && inList) parts.push("</ul>");
    inList = isItem;
    if (isItem) parts.push(`<li>${inline(line.replace(BULLET, ""))}</li>`);
    else if (line.trim()) parts.push(`<p>${inline(line.trim())}</p>`);
  }
  if (inList) parts.push("</ul>");
  return parts.join("");
}

/** Incremental Server-Sent Events parser: feed it text chunks as they arrive
 * (split anywhere), it calls onEvent({event, data}) per complete event. */
export function createSseParser(onEvent) {
  let buffer = "";
  return (chunk) => {
    buffer += chunk.replace(/\r\n/g, "\n");
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const event = parseBlock(block);
      if (event) onEvent(event);
    }
  };
}

function parseBlock(block) {
  let event = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return null; // comments (": ping") and empty blocks
  try {
    return { event, data: JSON.parse(data.join("\n")) };
  } catch {
    return null;
  }
}

export function ageMinutes(capturedAt, now = Date.now()) {
  return Math.max(0, Math.floor((now - Date.parse(capturedAt)) / MS_PER_MINUTE));
}
