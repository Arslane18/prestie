// Companion window: streamed chat + live character banner.
import { ageMinutes, createSseParser, renderMarkdown } from "./format.js";

const API_HEADERS = {
  "Content-Type": "application/json",
  "X-Prestie-Client": "1", // required by the API on every POST (anti-CSRF)
};
const STALE_AFTER_MINUTES = 30;
const AGE_REFRESH_MS = 30_000;

const $ = (id) => document.getElementById(id);
const log = $("log");
const question = $("question");
let character = null; // last state received, re-rendered as it ages

// --- character banner -----------------------------------------------------------

function renderCharacter() {
  if (!character) return;
  const c = character;
  $("char-name").textContent = c.character;
  $("char-class").textContent = `${c.class_name} niv. ${c.level}`;
  const spec = c.spec?.name ?? "sans spécialisation";
  const hero = c.hero_talent ? ` · ${c.hero_talent}` : "";
  const coverage = c.covered_by_knowledge_base
    ? ""
    : " · guides : DK Sang uniquement";
  $("char-spec").textContent = spec + hero + coverage;
  $("char-quest").textContent = c.active_quest
    ? `Quête suivie : ${c.active_quest.title ?? c.active_quest.id}`
    : "Aucune quête suivie";
  const age = ageMinutes(c.captured_at);
  const stale = age >= STALE_AFTER_MINUTES;
  setStatus(
    stale
      ? `État d'il y a ${age} min : tape /prestie sync en jeu pour l'actualiser.`
      : `État d'il y a ${age} min.`,
    stale ? "warn" : "",
  );
}

function setStatus(text, kind) {
  const status = $("char-status");
  status.hidden = !text;
  status.textContent = text;
  status.className = `status ${kind}`.trim();
}

async function loadCharacter() {
  try {
    const response = await fetch("/api/character");
    const body = await response.json();
    if (body.success) {
      character = body.data;
      renderCharacter();
    } else {
      setStatus(body.error, "error");
    }
  } catch {
    setStatus("Serveur Prestie injoignable.", "error");
  }
}

function watchCharacter() {
  // EventSource reconnects on its own if the server restarts.
  const source = new EventSource("/api/character/stream");
  source.addEventListener("state", (event) => {
    character = JSON.parse(event.data);
    renderCharacter();
  });
  source.addEventListener("error", (event) => {
    if (event.data) setStatus(JSON.parse(event.data).message, "error");
  });
}

// --- conversation ---------------------------------------------------------------

function addBubble(className) {
  $("empty")?.remove();
  const bubble = document.createElement("div");
  bubble.className = `bubble ${className}`;
  log.append(bubble);
  return bubble;
}

function scrollToEnd() {
  log.scrollTop = log.scrollHeight;
}

/** One assistant answer: collapsible steps (tool calls, notes) + the text. */
function createTurn() {
  const bubble = addBubble("assistant");
  const steps = document.createElement("details");
  steps.className = "steps";
  steps.hidden = true;
  const summary = document.createElement("summary");
  const list = document.createElement("ol");
  steps.append(summary, list);
  const answer = document.createElement("div");
  answer.className = "answer pending";
  bubble.append(steps, answer);

  return {
    addStep(text, kind) {
      const item = document.createElement("li");
      item.textContent = text;
      if (kind === "note") item.className = "note";
      list.append(item);
      steps.hidden = false;
      summary.textContent = text; // the latest step, while it runs
      scrollToEnd();
    },
    showAnswer(text) {
      answer.innerHTML = renderMarkdown(text); // escaped inside renderMarkdown
      scrollToEnd();
    },
    finish(reply) {
      answer.classList.remove("pending");
      answer.innerHTML = renderMarkdown(reply.text);
      const count = list.children.length;
      summary.textContent = `${count} étape${count > 1 ? "s" : ""}`;
      if (reply.truncated) addNotice(bubble, "Réponse tronquée (limite de longueur).");
      scrollToEnd();
    },
    fail(message) {
      answer.classList.remove("pending");
      bubble.classList.add("error");
      answer.textContent = message;
      scrollToEnd();
    },
  };
}

function addNotice(bubble, text) {
  const notice = document.createElement("p");
  notice.className = "notice";
  notice.textContent = text;
  bubble.append(notice);
}

async function ask(text) {
  setBusy(true);
  addBubble("user").textContent = text;
  const turn = createTurn();
  // Text streamed since the last tool call: an interim note until the answer.
  let roundText = "";
  let finished = false;
  const handle = ({ event, data }) => {
    if (event === "text") {
      roundText += data.text;
      turn.showAnswer(roundText);
    } else if (event === "tool") {
      if (roundText.trim()) turn.addStep(roundText.trim(), "note");
      roundText = "";
      turn.showAnswer("");
      turn.addStep(data.label, "tool");
    } else if (event === "restart") {
      roundText = "";
      turn.showAnswer("");
      turn.addStep("Autre modèle : la réponse reprend", "note");
    } else if (event === "done") {
      finished = true;
      turn.finish(data);
    } else if (event === "error") {
      finished = true;
      turn.fail(data.message);
    }
  };
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: API_HEADERS,
      body: JSON.stringify({ question: text }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new Error(body?.error ?? `HTTP ${response.status}`);
    }
    const feed = createSseParser(handle);
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      feed(value);
    }
    if (!finished) turn.fail("La réponse s'est interrompue.");
  } catch (error) {
    turn.fail(`Impossible de joindre le serveur Prestie (${error.message}).`);
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  $("send").disabled = busy;
  $("reset").disabled = busy;
  question.disabled = busy;
  if (!busy) question.focus();
}

async function resetConversation() {
  const response = await fetch("/api/reset", {
    method: "POST",
    headers: API_HEADERS,
  }).catch(() => null);
  if (!response?.ok) return;
  log.replaceChildren();
  const empty = document.createElement("p");
  empty.id = "empty";
  empty.className = "empty";
  empty.textContent = "Nouvelle conversation.";
  log.append(empty);
}

// --- wiring ---------------------------------------------------------------------

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text) return;
  question.value = "";
  ask(text);
});

question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});

$("reset").addEventListener("click", resetConversation);

loadCharacter();
watchCharacter();
setInterval(renderCharacter, AGE_REFRESH_MS);
question.focus();
