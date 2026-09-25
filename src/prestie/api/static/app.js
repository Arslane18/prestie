// Companion window: streamed chat, live character card, native window controls.
import {
  ageMinutes,
  classColor,
  createSseParser,
  renderMarkdown,
  stepLabel,
} from "./format.js";

const API_HEADERS = {
  "Content-Type": "application/json",
  "X-Prestie-Client": "1", // required by the API on every POST (anti-CSRF)
};
const STALE_AFTER_MINUTES = 30;
const AGE_REFRESH_MS = 30_000;
const MAX_INPUT_HEIGHT_PX = 140;

const $ = (id) => document.getElementById(id);
const log = $("log");
const question = $("question");
let character = null; // last state received, re-rendered as it ages
let busy = false;

// --- character card -------------------------------------------------------------

function renderCharacter() {
  if (!character) return;
  const c = character;
  const color = classColor(c.class_token);
  document.documentElement.style.setProperty("--class", color ?? "var(--gold)");
  $("char-name").textContent = `${c.character} · ${c.realm}`;
  $("char-level").hidden = false;
  $("char-level").textContent = `Niv. ${c.level}`;
  const spec = c.spec?.name ?? "Sans spécialisation";
  const hero = c.hero_talent ? ` · ${c.hero_talent}` : "";
  $("char-spec").textContent = `${c.class_name} · ${spec}${hero}`;
  $("char-quest").hidden = !c.active_quest;
  $("char-quest-title").textContent = c.active_quest?.title ?? "";
  renderAge(c);
}

function renderAge(c) {
  const age = ageMinutes(c.captured_at);
  const when = age < 1 ? "à l'instant" : `il y a ${formatDuration(age)}`;
  const status = $("char-status");
  status.hidden = false;
  status.replaceChildren(`Exporté ${when}`);
  status.className = "status";
  if (age >= STALE_AFTER_MINUTES) {
    status.className = "status warn";
    const command = document.createElement("code");
    command.textContent = "/prestie sync";
    status.append(" · ", command, " en jeu");
  }
  if (!c.covered_by_knowledge_base) {
    status.append(
      c.class_guides?.length
        ? " · pas encore de spé : guides de montée de niveau"
        : " · classe non couverte par les guides",
    );
  }
}

function formatDuration(minutes) {
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  return hours < 24 ? `${hours} h` : `${Math.floor(hours / 24)} j`;
}

function showCharacterError(message) {
  const status = $("char-status");
  status.hidden = false;
  status.className = "status error";
  status.textContent = message;
}

async function loadCharacter() {
  try {
    const body = await (await fetch("/api/character")).json();
    if (body.success) {
      character = body.data;
      renderCharacter();
    } else {
      showCharacterError(body.error);
    }
  } catch {
    showCharacterError("Serveur Prestie injoignable.");
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
    // Our "error" events carry data; the browser's connection errors do not.
    if (event.data) showCharacterError(JSON.parse(event.data).message);
  });
}

// --- conversation ---------------------------------------------------------------

function showEmptyState() {
  log.replaceChildren($("empty-state").content.cloneNode(true));
  for (const button of log.querySelectorAll(".suggestion")) {
    // Normalize the non-breaking spaces used for French typography.
    const text = button.textContent.replace(/\u00a0/g, " ");
    button.addEventListener("click", () => ask(text));
  }
}

function addMessage(kind) {
  log.querySelector(".empty")?.remove();
  const message = document.createElement("div");
  message.className = `message ${kind}`;
  log.append(message);
  return message;
}

function scrollToEnd() {
  log.scrollTop = log.scrollHeight;
}

/** One assistant answer: folded steps (tool calls, interim notes) + the text. */
function createTurn() {
  const message = addMessage("assistant");
  const steps = document.createElement("details");
  steps.className = "steps running";
  steps.hidden = true;
  const summary = document.createElement("summary");
  const list = document.createElement("ol");
  steps.append(summary, list);
  const answer = document.createElement("div");
  answer.className = "answer pending";
  message.append(steps, answer);

  return {
    addStep(text, kind) {
      const label = stepLabel(text);
      const item = document.createElement("li");
      item.textContent = label;
      if (kind === "note") item.className = "note";
      list.append(item);
      steps.hidden = false;
      summary.textContent = label; // the current step, while it runs
      scrollToEnd();
    },
    showAnswer(text) {
      answer.innerHTML = renderMarkdown(text); // escaped inside renderMarkdown
      scrollToEnd();
    },
    finish(reply) {
      answer.classList.remove("pending");
      steps.classList.remove("running");
      answer.innerHTML = renderMarkdown(reply.text);
      const count = list.children.length;
      summary.textContent = `${count} étape${count > 1 ? "s" : ""}`;
      if (reply.truncated) addNotice(message, "Réponse tronquée (limite de longueur).");
      scrollToEnd();
    },
    fail(text) {
      answer.classList.remove("pending");
      steps.classList.remove("running");
      message.classList.add("error");
      answer.textContent = text;
      scrollToEnd();
    },
  };
}

function addNotice(message, text) {
  const notice = document.createElement("p");
  notice.className = "notice";
  notice.textContent = text;
  message.append(notice);
}

async function ask(text) {
  if (busy) return;
  setBusy(true);
  addMessage("user").textContent = text;
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

function setBusy(value) {
  busy = value;
  $("send").disabled = value;
  $("reset").disabled = value;
  question.disabled = value;
  if (!value) question.focus();
}

async function resetConversation() {
  const response = await fetch("/api/reset", {
    method: "POST",
    headers: API_HEADERS,
  }).catch(() => null);
  if (response?.ok) showEmptyState();
}

function autoGrow() {
  question.style.height = "auto";
  question.style.height = `${Math.min(question.scrollHeight, MAX_INPUT_HEIGHT_PX)}px`;
}

// --- native window (pywebview) --------------------------------------------------

function enableWindowControls() {
  const api = window.pywebview?.api;
  if (!api) return;
  document.querySelector(".window-controls").hidden = false;
  const pin = $("pin");
  pin.addEventListener("click", async () => {
    const onTop = await api.set_on_top(pin.getAttribute("aria-pressed") !== "true");
    pin.setAttribute("aria-pressed", String(onTop));
    pin.classList.toggle("active", onTop);
  });
  $("minimize").addEventListener("click", () => api.minimize());
  $("close").addEventListener("click", () => api.close());
}

// --- wiring ---------------------------------------------------------------------

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text) return;
  question.value = "";
  autoGrow();
  ask(text);
});

question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});
question.addEventListener("input", autoGrow);
$("reset").addEventListener("click", resetConversation);
// pywebview injects window.pywebview after the page loads, then fires this.
window.addEventListener("pywebviewready", enableWindowControls);
enableWindowControls();

showEmptyState();
loadCharacter();
watchCharacter();
setInterval(() => character && renderAge(character), AGE_REFRESH_MS);
question.focus();
