import re
import json
import io
import logging
from typing import List, Dict
import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from config import OPENAI_API_KEY, OPENAI_API_BASE_URL

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Singapore Compliance Chatbot")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize OpenAI client (using base_url if provided)
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE_URL)

# Global configuration for embedding dimension
vector_dim = 768  # For "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Initialize the Sentence Transformer model
embedding_model = SentenceTransformer("./models/multi-qa-mpnet-base-dot-v1")

# Initialize a persistent Chroma client with a persist directory.
chroma_client = chromadb.PersistentClient(path="./db_chroma", settings=Settings(persist_directory="./db_chroma"))
collection = chroma_client.create_collection(name="pdpa_collection", get_or_create=True)

# Global list to hold our document metadata.
# Each document is a dict with keys: id, content, section_title, doc_type.
indexed_docs: List[Dict] = []

def extract_text_and_tables_from_pdf(file_bytes: bytes) -> (str, List[str]):
    pages = []
    full_text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""

            tables = page.extract_tables()
            for table in tables:
                table_text = "\n".join(
                    ["\t".join([str(cell) if cell is not None else "" for cell in row]) for row in table if row]
                )
                text += "\n\n[Extracted Table]\n" + table_text + "\n"

            full_text += text + "\n\n---PAGE BREAK---\n\n"
            pages.append(text)
    return full_text.strip(), pages

def llm_extract_section_titles_by_page(full_text: str) -> List[Dict]:
    """
    Uses an LLM to extract a list of section headings along with the page number 
    (1-indexed) where each section starts. The LLM is instructed to output exactly one valid
    JSON array of objects with keys "section_title" and "start_page".
    """
    prompt = "Document excerpt:\n" + full_text[:3000]+"""You are given the text of a legal policy document with page breaks indicated by '---PAGE BREAK---'.\nIdentify all the section headings (including appendix and annex) exactly as they appear in the document's table of contents, and for each heading, return the page number (starting from 1) where that section begins. Output exactly one valid JSON array of objects and nothing else. For example:\n'[{"section_title": "NOTICE TO BANKS", "start_page": 2}, {"section_title": "COMPUTATION OF TOTAL DEBT SERVICING RATIO FOR PROPERTY LOANS", "start_page": 4}]\n\n'"""
    print(prompt)
    try:
        response = client.chat.completions.create(
            model="qwen2.5",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            n=1,
        )
        raw_output = response.choices[0].message.content.strip()
        logging.info(f"LLM extract output: {raw_output}")
        # Use regex to extract the first JSON array from the output.
        match = re.search(r'(\[.*\])', raw_output, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in output")
        json_str = match.group(1)
        section_info = json.loads(json_str)
        if isinstance(section_info, dict):  # in case a single object is returned
            section_info = [section_info]
        return section_info
    except Exception as e:
        logging.error(f"Error extracting section titles by page: {e}")
        return []

def split_text_by_section_pages(pages: List[str], section_info: List[Dict]) -> List[Dict]:
    """
    Splits the full text (provided as a list of pages) into sections based on the extracted section info.
    Each section in section_info has "section_title" and "start_page".
    Returns a list of dicts with keys 'section_title' and 'content'.
    """
    # Sort section_info by start_page (assumes pages are 1-indexed)
    sections_sorted = sorted(section_info, key=lambda s: s.get("start_page", 1))
    result = []
    num_pages = len(pages)
    for i, sec in enumerate(sections_sorted):
        start_page = sec.get("start_page", 1)
        end_page = sections_sorted[i+1].get("start_page") if i+1 < len(sections_sorted) else num_pages + 1
        chunk = "\n\n".join(pages[start_page - 1 : end_page - 1]).strip()
        result.append({"section_title": sec.get("section_title", "Unknown"), "content": chunk})
    return result

def generate_qa_pairs(section: Dict) -> List[Dict]:
    """
    Uses the LLM to generate synthetic QA pairs from a section.
    Returns a list of dicts with keys 'question', 'answer', and 'section'.
    """
    prompt = (
        f"Generate three question and answer pairs from the following policy section "
        f"that a compliance team might ask. Include the section reference in each answer.\n\n"
        f"Section Title: {section['section_title']}\n\n"
        f"Section Content:\n{section['content']}\n\n"
        "Format the response exactly as:\nQuestion: <question>\nAnswer: <answer>\n"
    )
    try:
        response = client.chat.completions.create(
            model="qwen2.5",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            n=1,
        )
        generated_text = response.choices[0].message.content.strip()
        qa_pairs = []
        for pair in generated_text.split("\n\n"):
            if "Question:" in pair and "Answer:" in pair:
                parts = pair.split("Answer:")
                question_part = parts[0].split("Question:")
                if len(question_part) < 2:
                    continue
                question = question_part[1].strip()
                answer = parts[1].strip()
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "section": section["section_title"]
                })
        return qa_pairs
    except Exception as e:
        logging.error(f"Error generating QA pairs: {e}")
        return []

def compute_embeddings(texts: List[str]) -> np.ndarray:
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings

def add_to_chroma_collection(docs: List[Dict]):
    if not docs:
        return
    ids = [doc.get("id", "") for doc in docs]
    contents = [doc.get("content", "") for doc in docs]
    metadatas = [
        {"doc_id": doc.get("id", ""), "section_title": doc.get("section_title", ""), "doc_type": doc.get("doc_type", "")}
        for doc in docs
    ]
    embeddings = compute_embeddings(contents).tolist()
    collection.add(
        ids=ids,
        documents=contents,
        metadatas=metadatas,
        embeddings=embeddings
    )

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF, extracts its text and pages, uses LLM to extract section titles with page numbers,
    splits the document into sections based on these pages, generates QA pairs for each section, 
    computes embeddings, and indexes all documents into the persistent Chroma collection.
    """
    global indexed_docs
    try:
        file_bytes = await file.read()
        full_text, pages = extract_text_and_tables_from_pdf(file_bytes)
        section_info = llm_extract_section_titles_by_page(full_text)
        if not section_info:
            raise ValueError("No section titles could be extracted from the document.")
        sections = split_text_by_section_pages(pages, section_info)
        docs_to_index = []
        doc_id = 0
        for section in sections:
            docs_to_index.append({
                "id": f"doc_{doc_id}",
                "content": section["content"],
                "section_title": section["section_title"],
                "doc_type": "section"
            })
            doc_id += 1
            qa_pairs = generate_qa_pairs(section)
            for qa in qa_pairs:
                combined = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
                docs_to_index.append({
                    "id": f"doc_{doc_id}",
                    "content": combined,
                    "section_title": qa["section"],
                    "doc_type": "qa"
                })
                doc_id += 1
        indexed_docs = docs_to_index
        # Clear existing collection.
        existing = collection.get(include=["metadatas"])
        if existing.get("metadatas") and existing["metadatas"][0]:
            ids_to_delete = []
            for meta in existing["metadatas"][0]:
                # If meta is a dict, extract doc_id; if it's a str, assume it's the id.
                if isinstance(meta, dict):
                    doc_id = meta.get("doc_id")
                    if doc_id:
                        ids_to_delete.append(doc_id)
                elif isinstance(meta, str):
                    ids_to_delete.append(meta)
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
        add_to_chroma_collection(indexed_docs)
        return JSONResponse(content={
            "message": "PDF processed and indexed successfully",
            "num_sections": len(sections),
            "num_documents": len(indexed_docs)
        })
    except Exception as e:
        logging.error(f"Error in /upload_pdf: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def retrieve_relevant_docs(query: str, top_k: int = 5) -> List[Dict]:
    query_embedding = compute_embeddings([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances"])
    retrieved = []
    for metadata in results["metadatas"][0]:
        doc_id = metadata.get("doc_id")
        for doc in indexed_docs:
            if doc["id"] == doc_id:
                retrieved.append(doc)
                break
    return retrieved

def generate_answer(query: str) -> str:
    retrieved = retrieve_relevant_docs(query, top_k=5)
    context = ""
    for idx, doc in enumerate(retrieved):
        section = doc.get("section_title", "Unknown Section")
        content = doc.get("content", "")
        excerpt = content[:20000] + ("..." if len(content) > 500 else "")
        context += f"\nDocument {idx+1} (Section: {section}):\n{excerpt}\n"
    prompt = "You are a helpful legal compliance assistant for a bank. Answer the following question using only the provided legal policy excerpts. Include source citations using the section information (e.g. '(Section: 3)').\n\nRelevant Documents:" + context + "\n\n"+ f"Question: {query}\n\n"+"Provide a comprehensive, natural language answer with embedded source citations."
    
    try:
        response = client.chat.completions.create(
            model="qwen2.5",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            n=1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return f"Error generating answer: {e}"

@app.get("/generate")
def generate(q: str):
    try:
        answer = generate_answer(q)
        retrieved_docs = retrieve_relevant_docs(q, top_k=5)
        docs_info = [{"section": doc["section_title"], "doc_type": doc["doc_type"],
                      "excerpt": doc["content"][:200]} for doc in retrieved_docs]
        return JSONResponse(content={
            "query": q,
            "answer": answer,
            "retrieved_documents": docs_info
        })
    except Exception as e:
        logging.error(f"Error in /generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
def list_collections():
    collections = chroma_client.list_collections()
    coll_names = [coll.name for coll in collections]
    return {"collections": coll_names}

@app.get("/clauses")
def get_clauses():
    try:
        # Query all documents from Chroma (not just what's in memory)
        result = collection.get(include=["documents", "metadatas"])

        # Flatten out results
        docs = []
        for i, doc_text in enumerate(result["documents"]):
            metadata = result["metadatas"][i]
            docs.append({
                "id": metadata.get("doc_id"),
                "content": doc_text,
                "section_title": metadata.get("section_title", "Unknown"),
                "doc_type": metadata.get("doc_type", "unknown")
            })

        # Group by section title
        grouped = {}
        for doc in docs:
            section = doc["section_title"]
            if section not in grouped:
                grouped[section] = {"section_content": "", "qa_pairs": []}
            if doc["doc_type"] == "section":
                grouped[section]["section_content"] = doc["content"]
            elif doc["doc_type"] == "qa":
                grouped[section]["qa_pairs"].append(doc["content"])

        return JSONResponse(content={
            "clauses": [
                {
                    "section_title": section,
                    "section_content": data["section_content"],
                    "qa_pairs": data["qa_pairs"]
                } for section, data in grouped.items()
            ]
        })
    except Exception as e:
        logging.error(f"Error loading clauses from Chroma: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)