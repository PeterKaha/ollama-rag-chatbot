# Ollama RAG Chatbot

Ein vollständig **lokaler** RAG-Chatbot (Retrieval-Augmented Generation) der deine eigenen Dokumente durchsucht und mit einem lokalen LLM via [Ollama](https://ollama.com) beantwortet. Keine Cloud, keine API-Kosten.

## Architektur

```
Nutzer-Frage
     │
     ▼
┌─────────────┐    ┌──────────────────┐    ┌───────────────┐
│  Chatbot    │───▶│  RAG Pipeline    │───▶│  Ollama LLM   │
│  (CLI)      │    │  (Retrieval +    │    │  (Generation) │
└─────────────┘    │   Generation)    │    └───────────────┘
                   └────────┬─────────┘
                            │ Retrieval
                            ▼
                   ┌──────────────────┐    ┌───────────────┐
                   │   Chroma         │◀───│    Ollama     │
                   │   Vector Store   │    │   Embeddings  │
                   └──────────────────┘    └───────────────┘
                            ▲
                            │ Indexierung
                   ┌──────────────────┐
                   │  Document Loader │
                   │  (.txt .pdf .md) │
                   └──────────────────┘
                            ▲
                       ./data/
```

## Voraussetzungen

- Python 3.10+
- [Ollama](https://ollama.com) installiert und gestartet

## Setup

### 1. Ollama starten

```bash
ollama serve
```

### 2. Modelle herunterladen

```bash
# LLM für Antworten
ollama pull llama3.2

# Embedding-Modell für Vektorisierung
ollama pull nomic-embed-text
```

### 3. Projekt einrichten

```bash
# Virtuelle Umgebung erstellen
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# oder: .venv\Scripts\activate   # Windows

# Dependencies installieren
pip install -r requirements.txt

# Umgebungsvariablen konfigurieren
cp .env.example .env
# .env nach Bedarf anpassen
```

### 4. Dokumente hinzufügen

Lege deine Dateien in das `./data` Verzeichnis:

```bash
cp meine-dokumente/*.pdf ./data/
cp notizen/*.md ./data/
```

Unterstützte Formate: `.txt`, `.pdf`, `.md`

### 5. Chatbot starten

CLI-Modus:

```bash
python main.py
```

Web-Modus:

```bash
python main.py web
```

Danach ist das Frontend standardmaessig unter `http://127.0.0.1:8000` erreichbar.

Beim ersten Start werden alle Dokumente automatisch indexiert (Embeddings berechnet und in Chroma gespeichert). Folgestarts sind deutlich schneller, da bereits indexierte Quellen übersprungen werden.

## Verwendung

```
Du: Was steht in meinen Dokumenten über Kubernetes?

Assistent: Basierend auf den Dokumenten...
```

Im Webfrontend stellst du dieselben Fragen im Browser, siehst die Antwort direkt im Chatverlauf und darunter die verwendeten Quellen inklusive Relevanz und Dateipfad.

### Befehle im Chat

| Befehl    | Beschreibung                                     |
|-----------|--------------------------------------------------|
| `exit`    | Chatbot beenden                                  |
| `reindex` | Neue Dateien aus `./data` einlesen & indexieren  |
| `sources` | Quellen der letzten Antwort mit Relevanz anzeigen|
| `help`    | Alle Befehle anzeigen                            |

## Konfiguration (.env)

| Variable              | Standard                  | Beschreibung                              |
|-----------------------|---------------------------|-------------------------------------------|
| `OLLAMA_BASE_URL`     | `http://localhost:11434`  | Ollama API URL                            |
| `OLLAMA_LLM_MODEL`    | `llama3.2`                | Modell für Antwort-Generierung            |
| `OLLAMA_EMBED_MODEL`  | `nomic-embed-text`        | Modell für Embeddings                     |
| `CHROMA_PERSIST_DIR`  | `./chroma_db`             | Speicherort des Vector Stores             |
| `DATA_DIR`            | `./data`                  | Verzeichnis mit zu indexierenden Dateien  |
| `TOP_K_RESULTS`       | `5`                       | Anzahl relevanter Chunks als Kontext      |
| `WEB_HOST`            | `127.0.0.1`               | Host fuer das Webfrontend                 |
| `WEB_PORT`            | `8000`                    | Port fuer das Webfrontend                 |

## Projektstruktur

```
ollama-rag-chatbot/
├── src/
│   ├── document_loader.py   # Verzeichnis-Scan & Datei-Laden (.txt, .pdf, .md)
│   ├── embeddings.py        # Ollama-Embeddings (nomic-embed-text)
│   ├── vector_store.py      # Chroma Integration (persistent, mit Chunking)
│   ├── llm_client.py        # Ollama LLM Client (streaming)
│   ├── rag_pipeline.py      # RAG-Logik: Retrieval + Prompt + Generation
│   ├── chatbot.py           # CLI Chat Interface
│   ├── app_service.py       # Gemeinsame Initialisierung fuer CLI und Web
│   ├── web_app.py           # FastAPI Server + JSON API
│   └── web/
│       ├── templates/index.html
│       └── static/
│           ├── app.js
│           └── styles.css
├── data/                    # Eigene Dokumente hier ablegen
├── chroma_db/               # Vector Store (wird automatisch erstellt)
├── main.py                  # Entry Point
├── requirements.txt
├── .env.example
└── README.md
```

## Alternativer LLM

Jedes in Ollama verfügbare Modell kann verwendet werden:

```bash
ollama pull mistral
ollama pull gemma3
ollama pull deepseek-r1
```

Dann in `.env`:
```
OLLAMA_LLM_MODEL=mistral
```

## Web-API

Das Webfrontend nutzt drei Endpunkte:

- `GET /api/health` fuer Status, Modelle und Chunk-Anzahl
- `POST /api/chat` fuer Frage -> Antwort + Quellen
- `POST /api/reindex` fuer erneutes Einlesen des `./data` Verzeichnisses

## Alte Eintraege loeschen

Es gibt zwei Wege:

1. Kompletten Index leeren:

```bash
curl -X POST http://127.0.0.1:8000/api/clear-index
```

2. Nur Eintraege eines Dokuments loeschen (matcht auf Dateiname oder Quellpfad):

```bash
curl -X POST http://127.0.0.1:8000/api/delete-source \
     -H "Content-Type: application/json" \
     -d '{"source_query": "merkblatt-umzug-reisen_ba035665.pdf"}'
```

Danach bei Bedarf neu indexieren:

```bash
curl -X POST http://127.0.0.1:8000/api/reindex
```
