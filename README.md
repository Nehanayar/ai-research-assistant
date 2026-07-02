# 📚 AI Research Assistant

An AI-powered research assistant that lets you upload a research paper (PDF), ask questions about it with cited sources, or generate a quick summary — with a persistent memory of past conversations.

Built as a technical assignment prototype for **Webvory — AI Researcher / AI Innovation Engineer**.

## Live Demo
`<add your Streamlit Cloud URL here after deploying>`

## Features
- 📤 **Dynamic PDF upload** — index any research paper on the fly, no pre-processing step required
- 💬 **Q&A with source citations** — every answer shows which page(s) it was drawn from
- 📝 **Summarize mode** — one-click summary of the uploaded paper
- 🔁 **Visible multi-step pipeline** — retrieve → generate, shown as explicit steps (not a black-box single call)
- 🧠 **Persistent memory** — past Q&A is embedded and reused as context in future sessions

## Tech Stack
| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| LLM | Google Gemini (`gemini-2.5-flash`) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Memory | FAISS-backed conversation store |

## Architecture

```
User uploads PDF
      │
      ▼
PyPDFLoader → RecursiveCharacterTextSplitter → HuggingFace Embeddings → FAISS index (in-memory)
      │
      ▼
User asks a question
      │
      ▼
Step 1: Retrieve top-k chunks from PDF index + top-k from memory index
      │
      ▼
Step 2: Gemini generates answer using retrieved context
      │
      ▼
Answer shown with source page citations; Q&A pair embedded into memory index for future turns
```

## Project Structure
```
.
├── app.py              # Main Streamlit app (Q&A + Summarize modes)
├── index_builder.py     # Optional: pre-build a FAISS index from a fixed PDF
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`):
```
GEMINI_API_KEY=your_api_key_here
```

Run:
```bash
streamlit run app.py
```

Open the sidebar, upload a PDF, and start asking questions or generate a summary.

## Deliverables
- ✅ Working prototype (`app.py`)
- ✅ Documentation / recommendation report (`Research_Assistant_Report.docx`)
- ⬜ Demo video / walkthrough — record a 2–3 min screen capture showing upload → Q&A → summary
- ⬜ Screenshots — add to a `screenshots/` folder and link here
- ⬜ Deploy to Streamlit Cloud and add the live URL above

## Author
Neha Nayar
