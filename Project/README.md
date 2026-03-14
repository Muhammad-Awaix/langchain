# 🎬 YTChat — Chat with Any YouTube Video

A beautiful, industry-ready RAG web app that lets you chat with any YouTube video using Groq + LangChain + FAISS.

---

## ⚡ Quick Start (3 Steps)

### Step 1 — Clone / Download the project
Put all files in a folder called `ytchat/`. Structure should be:
```
ytchat/
├── app.py
├── requirements.txt
├── static/
│   └── index.html
└── README.md
```

### Step 2 — Install dependencies
Open your terminal inside the `ytchat/` folder and run:
```bash
pip install -r requirements.txt
```

### Step 3 — Run the server
```bash
uvicorn app:app --reload --port 8000
```

Then open your browser and go to:
```
http://localhost:8000
```

That's it! 🚀

---

## 🔑 How to Use

1. Get your **free Groq API key** from https://console.groq.com
2. Paste the API key in the sidebar
3. Paste any YouTube URL or video ID (e.g. `LPZh9BOjkQs`)
4. Click **Load Video** — wait ~10-15 seconds for the transcript to be processed
5. Start asking questions about the video!

---

## 🏗️ Architecture

```
Browser (index.html)
    │
    │  POST /load  →  fetch transcript → split → embed → FAISS index
    │  POST /chat  →  retrieve context → Groq LLM → answer
    │
FastAPI (app.py)
    ├── YouTubeTranscriptApi  — fetch video transcript
    ├── RecursiveCharacterTextSplitter  — split into chunks
    ├── HuggingFaceEmbeddings (all-MiniLM-L6-v2)  — embed chunks
    ├── FAISS  — vector store for similarity search
    └── ChatGroq (llama-3.3-70b-versatile)  — generate answers
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| RAG Framework | LangChain |
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| Transcript | youtube-transcript-api |
| Frontend | Vanilla HTML/CSS/JS |

---

## 📁 File Structure

```
ytchat/
├── app.py              ← FastAPI backend + RAG pipeline
├── requirements.txt    ← All Python dependencies
├── static/
│   └── index.html      ← Beautiful frontend UI
└── README.md
```

---

## ❓ Troubleshooting

**"Transcript not found"** → The video may not have captions enabled. Try a different video.

**"ModuleNotFoundError"** → Run `pip install -r requirements.txt` again.

**Slow first load** → The HuggingFace embedding model downloads on first use (~90MB). Subsequent runs are fast.

**Port already in use** → Change port: `uvicorn app:app --reload --port 8001`

---

## 🚀 Deploying Online (Free)

To share with others, deploy on **Render.com** (free):
1. Push this folder to a GitHub repo
2. Go to render.com → New Web Service → connect your repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Done — you get a public URL!

---

Built with ❤️ by dev66