/* manage.js – Dokumente-Seite */
"use strict";

// ── Health ──────────────────────────────────────────────────────────────────

async function fetchHealth() {
  const label = document.getElementById("health-label");
  const meta  = document.getElementById("health-meta");
  try {
    const res  = await fetch("/api/health");
    const data = await res.json();
    if (res.ok) {
      label.textContent = "Online";
      label.style.color = "var(--primary)";
      meta.textContent  = `${data.llm_model} · ${data.document_chunks} Chunks`;
    } else {
      label.textContent = "Fehler";
      label.style.color = "var(--error)";
    }
  } catch {
    label.textContent = "Nicht erreichbar";
    label.style.color = "var(--error)";
    meta.textContent  = "";
  }
}

// ── Sources table ────────────────────────────────────────────────────────────

async function refreshSources() {
  const container = document.getElementById("sources-container");
  const errEl     = document.getElementById("sources-error");
  errEl.textContent = "";
  container.innerHTML = '<p class="sources-empty">Lade…</p>';

  try {
    const res  = await fetch("/api/sources");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    renderSources(data.sources || []);
  } catch (err) {
    container.innerHTML = "";
    errEl.textContent = `Fehler beim Laden: ${err.message}`;
  }
}

function renderSources(sources) {
  const container = document.getElementById("sources-container");

  if (sources.length === 0) {
    container.innerHTML = '<p class="sources-empty">Keine indizierten Dokumente gefunden.</p>';
    return;
  }

  const table = document.createElement("table");
  table.className = "sources-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Dateiname</th>
        <th>Typ</th>
        <th>Chunks</th>
        <th>Seiten</th>
        <th>Status</th>
        <th></th>
      </tr>
    </thead>
    <tbody></tbody>
  `;

  const tbody = table.querySelector("tbody");
  for (const src of sources) {
    const tr = document.createElement("tr");
    const stale = !src.exists_on_disk;

    tr.innerHTML = `
      <td class="source-doc" title="${escHtml(src.source)}">${escHtml(src.filename)}</td>
      <td>${escHtml(src.type || "–")}</td>
      <td>${src.chunks}</td>
      <td>${src.page_count || "–"}</td>
      <td class="${stale ? "status-stale" : "status-ok"}">${stale ? "Veraltet" : "OK"}</td>
      <td>
        <button class="secondary-button compact delete-btn"
                data-source="${escHtml(src.source)}"
                type="button">Entfernen</button>
      </td>
    `;
    tbody.appendChild(tr);
  }

  container.innerHTML = "";
  container.appendChild(table);

  container.querySelectorAll(".delete-btn").forEach((btn) => {
    btn.addEventListener("click", () => deleteSource(btn.dataset.source, btn));
  });
}

async function deleteSource(source, btn) {
  if (!confirm(`Index-Einträge für "${source}" wirklich entfernen?`)) return;
  btn.disabled = true;
  try {
    const res  = await fetch("/api/delete-source", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ source_query: source }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    await Promise.all([refreshSources(), fetchHealth()]);
  } catch (err) {
    document.getElementById("sources-error").textContent = `Fehler: ${err.message}`;
    btn.disabled = false;
  }
}

// ── Cleanup ──────────────────────────────────────────────────────────────────

async function cleanupStale() {
  const btn    = document.getElementById("cleanup-button");
  const errEl  = document.getElementById("sources-error");
  errEl.textContent = "";
  btn.disabled = true;
  btn.classList.add("loading");
  try {
    const res  = await fetch("/api/cleanup-stale", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    if (data.removed_sources === 0) {
      errEl.style.color = "var(--muted)";
      errEl.textContent = "Keine veralteten Einträge gefunden.";
    } else {
      errEl.style.color = "var(--primary)";
      errEl.textContent = `${data.removed_sources} Quelle(n) bereinigt · ${data.chunks_deleted} Chunks entfernt.`;
    }
    await Promise.all([refreshSources(), fetchHealth()]);
  } catch (err) {
    errEl.textContent = `Fehler: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.classList.remove("loading");
  }
}

// ── Reindex ───────────────────────────────────────────────────────────────────

function showProgress(text, pct) {
  const box = document.getElementById("index-progress");
  const bar = document.getElementById("index-progress-bar");
  const label = document.getElementById("index-progress-text");
  box.hidden = false;
  bar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  label.textContent = text;
}

function hideProgress() {
  const box = document.getElementById("index-progress");
  if (box) {
    box.hidden = true;
  }
}

function streamReindex() {
  return new Promise((resolve) => {
    const errEl = document.getElementById("sources-error");
    const es = new EventSource("/api/reindex-stream");

    es.onmessage = (e) => {
      const event = JSON.parse(e.data);
      if (event.type === "start") {
        showProgress(`Indexiere ${event.documents} Dokument(e)...`, 0);
      } else if (event.type === "progress") {
        const pct = event.total_all > 0
          ? Math.round((event.total_done / event.total_all) * 100)
          : 0;
        showProgress(
          `${event.filename}: ${event.chunk_done}/${event.chunk_total} Chunks (${event.total_done}/${event.total_all} gesamt)`,
          pct,
        );
      } else if (event.type === "done") {
        showProgress("Fertig.", 100);
        errEl.style.color = "var(--primary)";
        errEl.textContent = `Fertig · ${event.chunks_added} neue Chunks · ${event.chunks_total} gesamt.`;
        setTimeout(hideProgress, 2500);
        es.close();
        Promise.all([refreshSources(), fetchHealth()]).then(resolve);
      } else if (event.type === "error") {
        errEl.style.color = "var(--error)";
        errEl.textContent = `Fehler: ${event.message}`;
        hideProgress();
        es.close();
        resolve();
      }
    };

    es.onerror = () => {
      errEl.style.color = "var(--error)";
      errEl.textContent = "Verbindung zum Fortschritts-Stream unterbrochen.";
      hideProgress();
      es.close();
      resolve();
    };
  });
}

async function reindex() {
  const btn   = document.getElementById("reindex-button");
  const errEl = document.getElementById("sources-error");
  errEl.textContent = "";
  errEl.style.color = "";
  btn.disabled = true;
  showProgress("Starte Indexierung...", 0);
  try {
    await streamReindex();
  } catch (err) {
    errEl.style.color = "var(--error)";
    errEl.textContent = `Fehler: ${err.message}`;
    hideProgress();
  } finally {
    btn.disabled = false;
  }
}

// ── Upload form ───────────────────────────────────────────────────────────────

function initUpload() {
  const form        = document.getElementById("upload-form");
  const fileInput   = document.getElementById("file-input");
  const drop        = document.getElementById("upload-drop");
  const nameDisplay = document.getElementById("file-name-display");
  const uploadBtn   = document.getElementById("upload-button");
  const resultEl    = document.getElementById("upload-result");

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    nameDisplay.textContent = file ? file.name : "";
    uploadBtn.disabled = !file;
  });

  // Drag-and-drop
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("drag-over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag-over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("drag-over");
    const file = e.dataTransfer?.files[0];
    if (file) {
      // Assign to input so FormData picks it up
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
      nameDisplay.textContent = file.name;
      uploadBtn.disabled = false;
    }
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) return;

    uploadBtn.disabled = true;
    uploadBtn.classList.add("loading");
    resultEl.textContent = "";
    resultEl.style.color = "";

    const fd = new FormData();
    fd.append("file", file);

    try {
      const res  = await fetch("/api/upload", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      resultEl.style.color = "var(--muted)";
      resultEl.textContent = `"${data.uploaded}" gespeichert. Indexiere…`;
      fileInput.value = "";
      nameDisplay.textContent = "";
      await streamReindex();
      resultEl.style.color = "var(--primary)";
      resultEl.textContent = `"${data.uploaded}" hochgeladen und indexiert.`;
    } catch (err) {
      resultEl.style.color = "var(--error)";
      resultEl.textContent = `Fehler: ${err.message}`;
    } finally {
      uploadBtn.disabled = true;
      uploadBtn.classList.remove("loading");
    }
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Boot ──────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  fetchHealth();
  refreshSources();
  initUpload();

  document.getElementById("refresh-button").addEventListener("click", refreshSources);
  document.getElementById("cleanup-button").addEventListener("click", cleanupStale);
  document.getElementById("reindex-button").addEventListener("click", reindex);
});
