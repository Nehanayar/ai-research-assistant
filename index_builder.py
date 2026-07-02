"""
Optional: bulk-index a PDF ahead of time instead of uploading through the UI.
Not required for normal use — app.py builds the index on the fly when you
upload a file in the sidebar. Use this only if you want a pre-built
'pdf_db' folder shipped with the app.

Usage:
    python index_builder.py sample.pdf
"""
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    db.save_local("pdf_db")

    print(f"PDF database created successfully from {pdf_path} ({len(chunks)} chunks)!")
