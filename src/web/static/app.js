const messages = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const healthLabel = document.getElementById("health-label");
const healthMeta = document.getElementById("health-meta");
const reindexButton = document.getElementById("reindex-button");
const template = document.getElementById("message-template");

function appendMessage({ role, text, sources = [], isError = false }) {
  const fragment = template.content.cloneNode(true);
  const article = fragment.querySelector(".message");
  const label = fragment.querySelector(".message-label");
  const body = fragment.querySelector(".message-body");
  const sourcesRoot = fragment.querySelector(".sources");

  article.classList.add(role === "user" ? "user-message" : "assistant-message");
  if (isError) {
    article.classList.add("error-message");
  }

  label.textContent = role === "user" ? "Du" : "Assistent";
  body.textContent = text;

  if (Array.isArray(sources) && sources.length > 0) {
    const label = document.createElement("div");
    label.className = "sources-label";
    label.textContent = "Quellen:";
    sourcesRoot.appendChild(label);
    sources.forEach((source) => {
      const filename = source.metadata?.filename || "Unbekannte Quelle";
      const page = source.metadata?.page ? `, Seite ${source.metadata.page}` : "";
      const link = document.createElement("a");
      link.className = "source-link";
      link.href = `/docs/${encodeURIComponent(filename)}`;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = `${filename}${page}`;
      sourcesRoot.appendChild(link);
    });
  }

  messages.appendChild(fragment);
  messages.scrollTop = messages.scrollHeight;
}

async function fetchHealth() {
  const response = await fetch("/api/health");
  const data = await response.json();
  healthLabel.textContent = data.ok ? "Bereit" : "Fehler";
  healthMeta.textContent = `${data.llm_model} | ${data.embed_model} | ${data.document_chunks} Chunks`;
}

async function submitQuestion(question) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question }),
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Unbekannter Fehler bei der Anfrage.");
  }
  return payload;
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  appendMessage({ role: "user", text: question });
  questionInput.value = "";

  const pending = template.content.cloneNode(true);
  const pendingArticle = pending.querySelector(".message");
  pendingArticle.classList.add("assistant-message");
  pending.querySelector(".message-label").textContent = "Assistent";
  const pendingBody = pending.querySelector(".message-body");
  pendingBody.textContent = "Antwort wird generiert";
  pendingBody.classList.add("loading");
  messages.appendChild(pending);
  messages.scrollTop = messages.scrollHeight;

  try {
    const result = await submitQuestion(question);
    const lastMessage = messages.lastElementChild;
    lastMessage.querySelector(".message-body").classList.remove("loading");
    lastMessage.querySelector(".message-body").textContent = result.answer;

    const sourcesRoot = lastMessage.querySelector(".sources");
    if (result.sources.length > 0) {
      const label = document.createElement("div");
      label.className = "sources-label";
      label.textContent = "Quellen:";
      sourcesRoot.appendChild(label);
    }
    result.sources.forEach((source) => {
      const filename = source.metadata?.filename || "Unbekannte Quelle";
      const page = source.metadata?.page ? `, Seite ${source.metadata.page}` : "";
      const link = document.createElement("a");
      link.className = "source-link";
      link.href = `/docs/${encodeURIComponent(filename)}`;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = `${filename}${page}`;
      sourcesRoot.appendChild(link);
    });
  } catch (error) {
    const lastMessage = messages.lastElementChild;
    lastMessage.classList.add("error-message");
    lastMessage.querySelector(".message-body").classList.remove("loading");
    lastMessage.querySelector(".message-body").textContent = error.message;
  }
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

reindexButton.addEventListener("click", async () => {
  reindexButton.disabled = true;
  reindexButton.textContent = "Indexiere...";
  try {
    const response = await fetch("/api/reindex", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Reindex fehlgeschlagen.");
    }
    appendMessage({
      role: "assistant",
      text: `Reindex abgeschlossen. ${payload.documents_loaded} Dokumente geladen, ${payload.chunks_added} neue Chunks indexiert, ${payload.chunks_total} Chunks insgesamt.`,
    });
    await fetchHealth();
  } catch (error) {
    appendMessage({ role: "assistant", text: error.message, isError: true });
  } finally {
    reindexButton.disabled = false;
    reindexButton.textContent = "Dokumente neu indexieren";
  }
});

fetchHealth().catch((error) => {
  healthLabel.textContent = "Fehler";
  healthMeta.textContent = error.message;
});