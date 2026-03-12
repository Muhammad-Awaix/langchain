<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=LangChain%20Journey&fontSize=50&fontColor=fff&animation=twinkling&fontAlignY=35&desc=From%20Zero%20to%20AI%20Engineer%20%7C%20Muhammad%20Awaix&descAlignY=55&descSize=16"/>

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=6C63FF&center=true&vCenter=true&width=600&lines=Building+AI+Apps+with+LangChain+%F0%9F%A6%9C;Prompts+%E2%86%92+Models+%E2%86%92+Chains+%E2%86%92+Agents;Pakistan's+Next+AI+Engineer+%F0%9F%9A%80;Learning+in+Public+%E2%9C%A8)](https://git.io/typing-svg)

<br/>

![Profile Views](https://komarev.com/ghpvc/?username=Muhammad-Awaix&color=6c63ff&style=for-the-badge&label=PROFILE+VIEWS)
[![GitHub followers](https://img.shields.io/github/followers/Muhammad-Awaix?style=for-the-badge&color=6c63ff)](https://github.com/Muhammad-Awaix)
[![GitHub stars](https://img.shields.io/github/stars/Muhammad-Awaix/langchain?style=for-the-badge&color=ff6b6b)](https://github.com/Muhammad-Awaix/langchain)

</div>

---

## 🧠 What is this repo?

This repository documents my **complete LangChain learning journey** — from writing my first prompt template to building full AI-powered pipelines. Every file here is a real experiment, a real error debugged, and a real concept understood.

> *"I don't just copy tutorials. I break things, fix them, and understand why."*

---

## 🗺️ Learning Roadmap

```
LangChain Fundamentals
│
├── 📌 Models          → HuggingFace, Groq, ChatModels
├── 📌 Prompts         → PromptTemplate, ChatPromptTemplate, partial_variables
├── 📌 Output Parsers  → StrOutputParser, JsonOutputParser, PydanticOutputParser
├── 📌 Chains          → Basic chains with | pipe operator
├── 📌 Runnables       → Parallel, Branch, Lambda, Passthrough
├── 📌 Document Loaders→ PyPDFLoader, DirectoryLoader
├── 📌 Text Splitters  → CharacterTextSplitter, RecursiveTextSplitter
└── 🔜 RAG Pipeline    → Coming soon...
```

---

## 📁 Repo Structure

```
langchain/
│
├── 📂 prompts/
│   ├── output_parsers.py       # StrOutput, JsonOutput, PydanticOutput
│   ├── op_p.py                 # Prompt + Parser experiments
│   └── ...
│
├── 📂 Chains/
│   ├── conditional_c.py        # RunnableBranch — if/else chains
│   ├── project.py              # 🌟 Full AI Feedback Analyzer project
│   └── ...
│
└── 📂 models/
    └── ...
```

---

## 🌟 Highlight Project — AI Customer Feedback Analyzer

> A real-world AI pipeline that automatically analyzes customer reviews

```python
chain = prompt | model | parser   # The beauty of LangChain
```

**What it does:**
- 🎯 Detects sentiment (Positive / Negative)
- 📝 Generates a one-line summary
- 🔑 Extracts keywords
- 💬 Auto-generates appropriate customer service response
- 📦 Returns everything as structured JSON

**Concepts used:** `RunnableParallel` · `RunnableBranch` · `RunnableLambda` · `RunnablePassthrough` · `PydanticOutputParser` · `ChatGroq`

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-F55036?style=for-the-badge&logo=groq&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

</div>

---

## 📚 Concepts Covered So Far

| Topic | Status | Key Files |
|-------|--------|-----------|
| PromptTemplate | ✅ Done | `prompts/` |
| ChatPromptTemplate | ✅ Done | `prompts/` |
| StrOutputParser | ✅ Done | `output_parsers.py` |
| JsonOutputParser | ✅ Done | `output_parsers.py` |
| PydanticOutputParser | ✅ Done | `op_p.py` |
| Basic Chains (`\|`) | ✅ Done | `Chains/` |
| RunnableParallel | ✅ Done | `project.py` |
| RunnableBranch | ✅ Done | `conditional_c.py` |
| RunnableLambda | ✅ Done | `project.py` |
| RunnablePassthrough | ✅ Done | `project.py` |
| Document Loaders | ✅ Done | `loaders/` |
| Text Splitters | 🔄 In Progress | — |
| Embeddings & VectorDB | 🔜 Next | — |
| RAG Pipeline | 🔜 Next | — |
| Agents & Tools | 🔜 Future | — |

---

## ⚡ Quick Start

```bash
# Clone the repo
git clone https://github.com/Muhammad-Awaix/langchain.git
cd langchain

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Install dependencies
pip install langchain langchain-core langchain-groq
pip install langchain-huggingface langchain-community
pip install python-dotenv pydantic pypdf

# Setup your API keys
cp .env.example .env
# Add your GROQ_API_KEY in .env
```

---

## 🔐 Environment Variables

Create a `.env` file in root:

```env
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
```

> ⚠️ **Never hardcode API keys in your code!** Always use `.env` files.

---

## 👨‍💻 About Me

<div align="center">

**Muhammad Awaix** — AI & Automation Enthusiast from Pakistan 🇵🇰

Learning LangChain through **CampusX GenAI Playlist**

Building in public · Breaking things · Understanding deeply

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Muhammad--Awaix-181717?style=for-the-badge&logo=github)](https://github.com/Muhammad-Awaix)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer"/>

*⭐ Star this repo if it helped you on your LangChain journey!*

</div>