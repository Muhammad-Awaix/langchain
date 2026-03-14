# ============================================================
# YTChat - Chat with any YouTube video
# Simple MVP version - easy to understand
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import re

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# ── Create the app ───────────────────────────────────────────
app = FastAPI()

# Allow browser to talk to this server
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Simple storage: saves loaded videos in memory while server is running
# Format: { "video_id": chain_object }
video_store = {}


# ── Request models (what the frontend sends us) ───────────────

class LoadRequest(BaseModel):
    video_id: str       # YouTube video ID or URL
    groq_api_key: str   # User's Groq API key

class ChatRequest(BaseModel):
    video_id: str       # Which video to ask about
    question: str       # The user's question
    groq_api_key: str   # User's Groq API key


# ── Helper: extract video ID from URL or return as-is ─────────

def get_video_id(text: str) -> str:
    # Handles URLs like: https://www.youtube.com/watch?v=ABC123
    # Or short URLs like: https://youtu.be/ABC123
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", text)
    if match:
        return match.group(1)
    return text.strip()  # assume it's already a bare ID like "ABC123"


# ── Helper: fetch transcript from YouTube ─────────────────────

def fetch_transcript(video_id: str) -> str:
    ytt = YouTubeTranscriptApi()
    chunks = ytt.fetch(video_id)                          # get list of caption chunks
    return " ".join(chunk.text for chunk in chunks)       # join into one big string


# ── Helper: build the RAG chain ───────────────────────────────
# RAG = Retrieval Augmented Generation
# Step 1: Split transcript into small chunks
# Step 2: Convert chunks to vectors (numbers) using embeddings
# Step 3: Store vectors in FAISS (fast search database)
# Step 4: When user asks a question, find the most relevant chunks
# Step 5: Send those chunks + question to Groq LLM for an answer

def build_rag_chain(transcript: str, groq_api_key: str):

    # Step 1: Split the transcript into chunks of 1000 characters
    # overlap=200 means chunks share 200 chars so context isn't lost at edges
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([transcript])

    # Step 2 & 3: Embed the chunks and store in FAISS vector database
    # all-MiniLM-L6-v2 is a small, fast, free embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Step 4: Retriever - finds top 4 most relevant chunks for any question
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Step 5: LLM - Groq is super fast and free
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.2,
    )

    # The prompt template - tells the LLM how to behave
    prompt = PromptTemplate(
        template="""You are a helpful assistant answering questions about a YouTube video.
Use ONLY the transcript context below. If the answer is not there, say so.

Context from video:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )

    # Combine retrieved docs into one string
    def join_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # Build the full chain:
    # Question → retrieve relevant chunks → format → prompt → LLM → text answer
    chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(join_docs),
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ── Route 1: Load a video ─────────────────────────────────────
# Frontend calls this first with a video ID
# We fetch the transcript and build the RAG chain

@app.post("/load")
async def load_video(req: LoadRequest):
    try:
        video_id = get_video_id(req.video_id)

        # If already loaded, skip re-processing (saves time)
        if video_id in video_store:
            return {"status": "ok", "message": "Already loaded!", "video_id": video_id}

        # Fetch transcript from YouTube
        transcript = fetch_transcript(video_id)

        if not transcript.strip():
            raise HTTPException(status_code=400, detail="Transcript is empty.")

        # Build the RAG chain and save it
        chain = build_rag_chain(transcript, req.groq_api_key)
        video_store[video_id] = chain

        return {
            "status": "ok",
            "video_id": video_id,
            "char_count": len(transcript),
            "preview": transcript[:200] + "...",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Route 2: Ask a question ───────────────────────────────────
# Frontend calls this with a question after the video is loaded

@app.post("/chat")
async def chat(req: ChatRequest):
    video_id = get_video_id(req.video_id)

    # Make sure video is loaded first
    if video_id not in video_store:
        raise HTTPException(status_code=404, detail="Video not loaded. Please load it first.")

    try:
        chain = video_store[video_id]
        answer = chain.invoke(req.question)

        # Remove <think>...</think> tags some models add internally
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Route 3: Serve the frontend HTML ─────────────────────────
# Instead of a separate static folder, we embed the HTML directly
# This way you only need ONE file (app.py) to run everything!

@app.get("/", response_class=HTMLResponse)
async def frontend():
    return HTML_PAGE   # defined below


# ============================================================
# FRONTEND HTML - embedded directly in Python
# No need for a separate static/ folder!
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>YTChat</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0a0a0f;
    --surface:  #111118;
    --surface2: #18181f;
    --border:   #ffffff0f;
    --border2:  #ffffff18;
    --red:      #ff3b3b;
    --red-dim:  #ff3b3b22;
    --red-glow: #ff3b3b55;
    --white:    #f0f0f5;
    --muted:    #6b6b80;
    --muted2:   #9090a8;
  }

  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--white);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Header ── */
  header {
    padding: 18px 28px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0a0a0fcc;
    backdrop-filter: blur(10px);
    position: sticky;
    top: 0;
    z-index: 10;
  }

  .logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.2rem;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .logo-dot {
    width: 30px; height: 30px;
    background: var(--red);
    border-radius: 7px;
    display: grid;
    place-items: center;
    font-size: 15px;
    box-shadow: 0 0 18px var(--red-glow);
  }

  .tag {
    font-size: 0.7rem;
    border: 1px solid var(--border2);
    padding: 3px 10px;
    border-radius: 100px;
    color: var(--muted2);
    font-family: 'DM Sans', sans-serif;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }

  /* ── Layout ── */
  main {
    flex: 1;
    display: grid;
    grid-template-columns: 320px 1fr;
    height: calc(100vh - 61px);
  }

  /* ── Sidebar ── */
  aside {
    border-right: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }

  .sidebar-block {
    padding: 22px;
    border-bottom: 1px solid var(--border);
  }

  .label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
    font-family: 'Syne', sans-serif;
  }

  /* ── Inputs ── */
  input[type=text], input[type=password] {
    width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 11px;
    padding: 10px 13px;
    color: var(--white);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.86rem;
    outline: none;
    margin-bottom: 10px;
    transition: border-color 0.2s, box-shadow 0.2s;
  }

  input:focus {
    border-color: var(--red);
    box-shadow: 0 0 0 3px var(--red-dim);
  }

  input::placeholder { color: var(--muted); }

  /* ── Button ── */
  button.primary {
    width: 100%;
    padding: 11px;
    background: var(--red);
    color: #fff;
    border: none;
    border-radius: 11px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.88rem;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 4px 20px var(--red-dim);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }

  button.primary:hover   { transform: translateY(-1px); box-shadow: 0 6px 28px var(--red-glow); }
  button.primary:active  { transform: none; }
  button.primary:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  /* ── Status messages ── */
  .status {
    font-size: 0.78rem;
    padding: 8px 11px;
    border-radius: 8px;
    margin-top: 10px;
    display: none;
  }
  .status.ok   { background:#1a3a1a; color:#5dde5d; border:1px solid #2d5a2d; display:block; }
  .status.err  { background:#3a1a1a; color:#de5d5d; border:1px solid #5a2d2d; display:block; }
  .status.info { background:#1a1a3a; color:#5d8ede; border:1px solid #2d3a5a; display:block; }

  /* ── Quick prompts ── */
  .quick-prompts { display: flex; flex-direction: column; gap: 7px; }

  .qp {
    padding: 8px 12px;
    border-radius: 9px;
    border: 1px solid var(--border2);
    background: var(--surface2);
    color: var(--muted2);
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.18s;
  }

  .qp:hover { border-color: var(--red); color: var(--white); background: var(--red-dim); }

  /* ── Chat area ── */
  .chat-area {
    display: flex;
    flex-direction: column;
    background: var(--bg);
  }

  /* ── Empty state ── */
  #emptyState {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    color: var(--muted);
    text-align: center;
    padding: 40px;
  }

  #emptyState .icon  { font-size: 50px; opacity: 0.25; animation: pulse 3s ease-in-out infinite; }
  #emptyState h2     { font-family:'Syne',sans-serif; font-size:1.25rem; color:var(--muted2); }
  #emptyState p      { font-size:0.85rem; line-height:1.6; max-width:280px; }

  /* ── Messages ── */
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px 28px;
    display: none;
    flex-direction: column;
    gap: 18px;
    scroll-behavior: smooth;
  }

  #messages::-webkit-scrollbar { width: 4px; }
  #messages::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

  .msg {
    display: flex;
    gap: 10px;
    max-width: 760px;
    animation: fadeUp 0.25s ease;
  }

  .msg.user      { align-self: flex-end; flex-direction: row-reverse; }
  .msg.assistant { align-self: flex-start; }

  .av {
    width: 30px; height: 30px;
    border-radius: 7px;
    display: grid;
    place-items: center;
    font-size: 13px;
    flex-shrink: 0;
    margin-top: 2px;
  }

  .msg.user .av      { background: var(--red); }
  .msg.assistant .av { background: var(--surface2); border: 1px solid var(--border2); }

  .bubble {
    padding: 12px 16px;
    border-radius: 13px;
    font-size: 0.88rem;
    line-height: 1.65;
    max-width: 600px;
  }

  .msg.user .bubble      { background: var(--red); color: #fff; border-bottom-right-radius: 3px; }
  .msg.assistant .bubble { background: var(--surface2); border: 1px solid var(--border); border-bottom-left-radius: 3px; }

  /* ── Typing dots ── */
  .typing { display:flex; gap:5px; padding:13px 16px; align-items:center; }
  .typing span { width:7px; height:7px; border-radius:50%; background:var(--muted); animation:bounce 1.2s infinite; }
  .typing span:nth-child(2) { animation-delay:0.2s; }
  .typing span:nth-child(3) { animation-delay:0.4s; }

  /* ── Input bar ── */
  .input-bar {
    padding: 16px 28px 20px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    gap: 10px;
    align-items: flex-end;
  }

  .input-bar textarea {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 11px;
    padding: 11px 15px;
    color: var(--white);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    resize: none;
    outline: none;
    min-height: 46px;
    max-height: 120px;
    line-height: 1.5;
    transition: border-color 0.2s, box-shadow 0.2s;
  }

  .input-bar textarea:focus { border-color: var(--red); box-shadow: 0 0 0 3px var(--red-dim); }
  .input-bar textarea::placeholder { color: var(--muted); }

  .send {
    width: 46px; height: 46px;
    background: var(--red);
    border: none;
    border-radius: 11px;
    color: #fff;
    font-size: 17px;
    cursor: pointer;
    display: grid;
    place-items: center;
    transition: all 0.2s;
    box-shadow: 0 4px 18px var(--red-dim);
    flex-shrink: 0;
  }

  .send:hover    { transform: translateY(-1px); box-shadow: 0 6px 24px var(--red-glow); }
  .send:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  /* ── Animations ── */
  @keyframes fadeUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
  @keyframes bounce { 0%,80%,100% { transform:scale(0.7); opacity:0.4; } 40% { transform:scale(1); opacity:1; } }
  @keyframes pulse  { 0%,100% { opacity:0.25; } 50% { opacity:0.4; } }

  .spinner { display:inline-block; width:13px; height:13px; border:2px solid #ffffff44; border-top-color:#fff; border-radius:50%; animation:spin 0.7s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }

  @media (max-width: 700px) {
    main { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
    aside { max-height: 280px; border-right: none; border-bottom: 1px solid var(--border); }
  }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-dot">▶</div>
    YTChat
  </div>
  <div class="tag">Groq + LangChain RAG</div>
</header>

<main>
  <!-- ── Sidebar ── -->
  <aside>
    <div class="sidebar-block">
      <div class="label">🔑 Groq API Key</div>
      <input type="password" id="apiKey" placeholder="gsk_..."/>
    </div>

    <div class="sidebar-block">
      <div class="label">📺 YouTube Video</div>
      <input type="text" id="videoInput" placeholder="Paste URL or video ID"/>
      <button class="primary" id="loadBtn" onclick="loadVideo()">
        <span id="loadTxt">Load Video</span>
      </button>
      <div class="status" id="loadStatus"></div>
    </div>

    <div class="sidebar-block" style="flex:1;">
      <div class="label">⚡ Quick Prompts</div>
      <div class="quick-prompts">
        <div class="qp" onclick="usePrompt(this)">Summarize this video</div>
        <div class="qp" onclick="usePrompt(this)">What are the key points?</div>
        <div class="qp" onclick="usePrompt(this)">List all topics covered</div>
        <div class="qp" onclick="usePrompt(this)">Any actionable takeaways?</div>
        <div class="qp" onclick="usePrompt(this)">Who is being discussed?</div>
      </div>
    </div>
  </aside>

  <!-- ── Chat ── -->
  <section class="chat-area">
    <div id="emptyState">
      <div class="icon">🎬</div>
      <h2>Chat with any YouTube video</h2>
      <p>Enter your Groq API key and a YouTube URL on the left, then start asking questions.</p>
    </div>

    <div id="messages"></div>

    <div class="input-bar">
      <textarea id="qInput" placeholder="Ask anything about the video… (Enter to send)"
        onkeydown="handleKey(event)" oninput="resize(this)" rows="1"></textarea>
      <button class="send" id="sendBtn" onclick="send()" title="Send">➤</button>
    </div>
  </section>
</main>

<script>
  // Tracks which video is currently loaded
  let currentVideoId = null;
  let busy = false;

  // ── Utilities ─────────────────────────────────────────────

  function setStatus(msg, type) {
    const el = document.getElementById('loadStatus');
    el.textContent = msg;
    el.className = 'status ' + type;
  }

  function resize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  }

  function usePrompt(el) {
    const ta = document.getElementById('qInput');
    ta.value = el.textContent;
    resize(ta);
    ta.focus();
  }

  function showChat() {
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('messages').style.display = 'flex';
  }

  function scrollDown() {
    const m = document.getElementById('messages');
    m.scrollTop = m.scrollHeight;
  }

  // ── Add a message bubble ──────────────────────────────────

  function addMsg(role, text) {
    showChat();
    const msgs = document.getElementById('messages');

    const wrap = document.createElement('div');
    wrap.className = 'msg ' + role;

    const av = document.createElement('div');
    av.className = 'av';
    av.textContent = role === 'user' ? '👤' : '🤖';

    const bub = document.createElement('div');
    bub.className = 'bubble';
    bub.textContent = text;

    wrap.appendChild(av);
    wrap.appendChild(bub);
    msgs.appendChild(wrap);
    scrollDown();
    return wrap;
  }

  // Animated typing dots while waiting for answer
  function addTyping() {
    const msgs = document.getElementById('messages');
    const wrap = document.createElement('div');
    wrap.className = 'msg assistant';
    wrap.id = 'typing';

    const av = document.createElement('div');
    av.className = 'av';
    av.textContent = '🤖';

    const bub = document.createElement('div');
    bub.className = 'bubble typing';
    bub.innerHTML = '<span></span><span></span><span></span>';

    wrap.appendChild(av);
    wrap.appendChild(bub);
    msgs.appendChild(wrap);
    scrollDown();
  }

  function removeTyping() {
    document.getElementById('typing')?.remove();
  }

  // ── Load video ────────────────────────────────────────────

  async function loadVideo() {
    const apiKey = document.getElementById('apiKey').value.trim();
    const videoInput = document.getElementById('videoInput').value.trim();

    if (!apiKey)      return setStatus('Enter your Groq API key first.', 'err');
    if (!videoInput)  return setStatus('Enter a YouTube URL or video ID.', 'err');

    // Disable button and show spinner
    const btn = document.getElementById('loadBtn');
    const txt = document.getElementById('loadTxt');
    btn.disabled = true;
    txt.innerHTML = '<span class="spinner"></span> Loading…';
    setStatus('Fetching transcript… please wait ~10 seconds.', 'info');

    try {
      const res = await fetch('/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoInput, groq_api_key: apiKey })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to load video');

      currentVideoId = data.video_id;
      setStatus('✅ Ready! Start asking questions.', 'ok');
      showChat();
      addMsg('assistant', `Video loaded! (${(data.char_count||0).toLocaleString()} chars of transcript). What would you like to know? 🎬`);

    } catch (err) {
      setStatus('❌ ' + err.message, 'err');
    } finally {
      btn.disabled = false;
      txt.textContent = 'Load Video';
    }
  }

  // ── Send a question ───────────────────────────────────────

  async function send() {
    if (busy) return;

    const apiKey   = document.getElementById('apiKey').value.trim();
    const ta       = document.getElementById('qInput');
    const question = ta.value.trim();

    if (!question) return;

    if (!currentVideoId) {
      addMsg('assistant', '⚠️ Load a YouTube video first using the sidebar.');
      return;
    }

    busy = true;
    document.getElementById('sendBtn').disabled = true;
    ta.value = '';
    resize(ta);

    addMsg('user', question);
    addTyping();

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: currentVideoId, question, groq_api_key: apiKey })
      });

      const data = await res.json();
      removeTyping();

      if (!res.ok) throw new Error(data.detail || 'Error getting answer');
      addMsg('assistant', data.answer);

    } catch (err) {
      removeTyping();
      addMsg('assistant', '❌ ' + err.message);
    } finally {
      busy = false;
      document.getElementById('sendBtn').disabled = false;
      ta.focus();
    }
  }
</script>
</body>
</html>
"""