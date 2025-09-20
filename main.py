# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from pypdf import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import os
import google.generativeai as genai
from typing import List

# --- Environment and Model Setup ---
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# --- LAZY LOADING IMPLEMENTATION ---
# Initialize to None. The model will only be loaded when the first request comes in.
embeddings = None
def get_embeddings():
    """Initializes and returns the embedding model, loading it only once."""
    global embeddings
    if embeddings is None:
        print("Lazy loading embedding model...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Model loaded.")
    return embeddings
# ---

app = FastAPI(
    title="Smart Doc Checker Agent API",
    description="An API that scans multiple docs, flags contradictions, and reports issues clearly.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Smart Doc Checker Agent API!"}

@app.post("/analyze/")
async def analyze_documents(files: List[UploadFile] = File(...)):
    if len(files) < 2:
        return JSONResponse(status_code=400, content={"message": "Please upload at least two documents to compare."})

    try:
        # Load the model only when this endpoint is called
        current_embeddings = get_embeddings()
        
        all_chunks = []
        for file in files:
            if file.content_type != "application/pdf": continue
            file_content = await file.read()
            pdf_reader = PdfReader(io.BytesIO(file_content))
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(text=text)
            
            for i, chunk_text in enumerate(chunks):
                doc = Document(page_content=chunk_text, metadata={"source": file.filename, "chunk_id": i})
                all_chunks.append(doc)

        if not all_chunks:
            return JSONResponse(status_code=400, content={"message": "Could not extract text from the provided PDFs."})

        vector_store = FAISS.from_documents(all_chunks, embedding=current_embeddings)

        conflicts = []
        checked_pairs = set()
        for i, doc1 in enumerate(all_chunks):
            similar_docs = vector_store.similarity_search(
                query=doc1.page_content, 
                k=5,
                filter=lambda metadata: metadata["source"] != doc1.metadata["source"]
            )
            for doc2 in similar_docs:
                pair_key = tuple(sorted((f"{doc1.metadata['source']}-{doc1.metadata['chunk_id']}", f"{doc2.metadata['source']}-{doc2.metadata['chunk_id']}")))
                if pair_key in checked_pairs: continue
                checked_pairs.add(pair_key)

                prompt = f"""
                You are a document compliance auditor. Analyze these two snippets. Do they contradict each other?

                Document 1 ('{doc1.metadata['source']}') says: "{doc1.page_content}"
                Document 2 ('{doc2.metadata['source']}') says: "{doc2.page_content}"

                If there is a clear contradiction, respond with "Conflict:" followed by a concise, one-sentence explanation.
                If there is no contradiction, respond ONLY with the words "No Conflict".
                """
                
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                response = model.generate_content(prompt)

                if "No Conflict" not in response.text:
                    conflicts.append({
                        "explanation": response.text.replace("Conflict:", "").strip(),
                        "doc1_source": doc1.metadata['source'],
                        "doc1_content": doc1.page_content,
                        "doc2_source": doc2.metadata['source'],
                        "doc2_content": doc2.page_content
                    })
        
        return {"conflicts": conflicts}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})