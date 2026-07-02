import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# -------------------- Setup --------------------
load_dotenv()
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API Key not found.")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.3
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

st.set_page_config(page_title="AI Research Assistant", page_icon="📚")
st.title("📚 AI Research Assistant")
st.caption("Upload a research paper, ask questions, or get a quick summary — with sources cited.")

# -------------------- Session State --------------------
if "memories" not in st.session_state:
    st.session_state.memories = []
if "pdf_db" not in st.session_state:
    st.session_state.pdf_db = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# -------------------- Memory DB (persists across sessions) --------------------
if os.path.exists("memory_db"):
    memory_db = FAISS.load_local("memory_db", embeddings, allow_dangerous_deserialization=True)
else:
    memory_db = FAISS.from_texts(["memory initialized"], embeddings)

# -------------------- Sidebar: PDF Upload --------------------
with st.sidebar:
    st.header("📄 Document")
    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

    if uploaded_file is not None and uploaded_file.name != st.session_state.doc_name:
        with st.spinner("Reading and indexing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            st.session_state.pdf_db = FAISS.from_documents(chunks, embeddings)
            st.session_state.doc_name = uploaded_file.name
            os.remove(tmp_path)

        st.success(f"Indexed: {uploaded_file.name} ({len(chunks)} chunks)")

    elif st.session_state.pdf_db is not None:
        st.info(f"Active document: {st.session_state.doc_name}")
    else:
        st.warning("Upload a PDF to get started.")

    st.divider()
    mode = st.radio("Mode", ["Ask a question", "Summarize the paper"])

# -------------------- Helper: Retrieve with sources --------------------
def retrieve_with_sources(query, k=4):
    if st.session_state.pdf_db is None:
        return "", []
    results = st.session_state.pdf_db.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    sources = [
        f"Page {doc.metadata.get('page', '?') + 1 if isinstance(doc.metadata.get('page'), int) else '?'}"
        for doc in results
    ]
    return context, sources

# -------------------- Mode: Summarize --------------------
if mode == "Summarize the paper":
    if st.button("Generate Summary", disabled=st.session_state.pdf_db is None):
        with st.spinner("Step 1/2: Retrieving key sections..."):
            # Pull a broad sample of chunks to represent the whole doc
            context, sources = retrieve_with_sources("main findings, methodology, conclusion", k=8)

        with st.spinner("Step 2/2: Summarizing..."):
            summary_prompt = f"""
Summarize the following research paper content in 150-200 words.
Cover: the core problem, the method/approach, and the key findings.

Content:
{context}
"""
            response = llm.invoke(summary_prompt)

        st.subheader("Summary")
        st.write(response.content)
        with st.expander("Sources used"):
            st.write(", ".join(sorted(set(sources))))

# -------------------- Mode: Q&A --------------------
else:
    question = st.chat_input("Ask a question about the document")

    if question:
        with st.spinner("Step 1/2: Retrieving relevant context..."):
            pdf_context, pdf_sources = retrieve_with_sources(question, k=4)
            memory_docs = memory_db.similarity_search(question, k=4)
            memory_context = "\n".join([doc.page_content for doc in memory_docs])

        with st.spinner("Step 2/2: Generating answer..."):
            prompt = f"""
You are a helpful research assistant.

User Memories:
{memory_context}

PDF Context:
{pdf_context}

Current Question:
{question}

Instructions:
- Use PDF context first.
- Use memories if relevant.
- If the answer is not available in the PDF, clearly say so.
"""
            response = llm.invoke(prompt)
            answer = response.content

        st.session_state.memories.append(("user", question, None))
        st.session_state.memories.append(("assistant", answer, pdf_sources))

        memory_db.add_documents([
            Document(page_content=f"User: {question}"),
            Document(page_content=f"Assistant: {answer}")
        ])
        memory_db.save_local("memory_db")

    # Display chat history with citations
    for entry in st.session_state.memories:
        role, message, sources = entry
        with st.chat_message(role):
            st.write(message)
            if sources:
                st.caption(f"Sources: {', '.join(sorted(set(sources)))}")
