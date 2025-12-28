from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import pandas as pd
import io
import base64
import os
from dotenv import load_dotenv
from model_utils import sentiment_map, aspect_map
from geo_utils import get_coordinates, fetch_hospitals, fallback_hospitals
from review_analysis import analyze_reviews, score_hospital
from visualizations import generate_wordcloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from collections import Counter
from groq import Groq
import numpy as np
import pickle
import uuid
import re
import hashlib
import time
import json
import asyncio

app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

try:
    load_dotenv()
except Exception:
    pass

# Environment variables
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"DEBUG: Loaded GROQ_API_KEY: {GROQ_API_KEY[:5] if GROQ_API_KEY else 'None'}")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store.pkl")

from sentence_transformers import SentenceTransformer
import warnings
# Suppress huggingface warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global RAG variables - ChromaDB
_embeddings_model = None
_local_embedding_model = None
_chroma_client = None
_chroma_collection = None
_embeddings_cache = {}
_last_api_call_time = 0  # Rate limiting
_min_api_interval = 1.0  # Minimum seconds between API calls

# ============================================================================
# CHROMADB RAG IMPLEMENTATION
# ============================================================================

import chromadb
from chromadb.config import Settings

def _init_embedding_model():
    """Initialize local embedding model"""
    global _local_embedding_model
    try:
        if _local_embedding_model is None:
            print("Loading local embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
            _local_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[OK] Local embedding model loaded")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load local embedding model: {e}")
        return False

# Groq Client Global
_groq_client = None

def _configure_groq():
    """Configure Groq API"""
    global _groq_client
    if GROQ_API_KEY:
        try:
            if not _groq_client:
                _groq_client = Groq(api_key=GROQ_API_KEY)
            print("[OK] Groq API configured")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to configure Groq: {e}")
            return False
    print("[WARNING] No Groq API key provided")
    return False

def _pick_groq_model():
    """Return preferred Groq model"""
    return "llama-3.3-70b-versatile"

def get_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding for text using local model"""
    if not text or not isinstance(text, str):
        return None
    
    # Check cache first
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash in _embeddings_cache:
        return _embeddings_cache[text_hash]
    
    try:
        # Initialize if needed
        if _local_embedding_model is None:
            _init_embedding_model()
            
        if _local_embedding_model:
            # Generate embedding
            embedding = _local_embedding_model.encode(text)
            
            # Ensure numpy array float32
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)
            else:
                embedding = embedding.astype(np.float32)
            
            # Cache it
            _embeddings_cache[text_hash] = embedding
            return embedding
            
    except Exception as e:
        print(f"Error getting embedding: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def get_query_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding for query using local model"""
    # Local model uses same logic for documents and queries
    return get_embedding(text)

def _init_vectorstore():
    """Initialize ChromaDB vector store"""
    global _chroma_client, _chroma_collection
    
    print("\n=== Initializing ChromaDB Vector Store ===")
    
    # Initialize local embedding model
    if not _init_embedding_model():
        print("[ERROR] Could not initialize embedding model")
    
    # Configure Groq (still needed for generation)
    _configure_groq()
    
    try:
        # Initialize persistent client
        db_path = os.path.join(os.getcwd(), "chroma_db")
        print(f"ChromaDB Path: {db_path}")
        
        _chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        # We manually handle embeddings, so using DefaultEmbeddingFunction is optional/redundant but harmless if we pass embeddings directly
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="hospital_reviews",
            metadata={"hnsw:space": "cosine"}
        )
        
        count = _chroma_collection.count()
        print(f"[OK] ChromaDB initialized. Collection 'hospital_reviews' has {count} documents.")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=== RAG Initialization Complete ===\n")
    return True

def add_documents(texts: List[str], metadatas: List[Dict], batch_size: int = 20):
    """Add documents to ChromaDB vector store"""
    global _chroma_collection
    
    if not texts or not _chroma_collection:
        return 0
    
    added_count = 0
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        
        batch_ids = []
        batch_embeddings = []
        valid_texts = []
        valid_metas = []
        
        for text, meta in zip(batch_texts, batch_metas):
            try:
                # Create unique ID
                doc_id = hashlib.sha1((str(meta.get('hospital_name', '')) + text).encode()).hexdigest()
                
                # Get embedding
                embedding = get_embedding(text)
                
                if embedding is not None:
                    batch_ids.append(doc_id)
                    batch_embeddings.append(embedding.tolist())
                    valid_texts.append(text)
                    valid_metas.append(meta)
                else:
                    print(f"[WARNING] Could not get embedding for text: {text[:50]}...")
            
            except Exception as e:
                print(f"Error preparing document: {e}")
                continue
        
        if valid_texts:
            try:
                _chroma_collection.upsert(
                    ids=batch_ids,
                    documents=valid_texts,
                    metadatas=valid_metas,
                    embeddings=batch_embeddings
                )
                added_count += len(valid_texts)
            except Exception as e:
                print(f"[ERROR] Failed to add batch to ChromaDB: {e}")
        
        print(f"[OK] Processed batch {i//batch_size + 1}, added {added_count} documents so far")
    
    return added_count

def search_documents(query: str, k: int = 8, hospital_names: Optional[List[str]] = None):
    """Search for similar documents using ChromaDB"""
    global _chroma_collection
    
    if not _chroma_collection:
        print("[ERROR] Chroma collection is not initialized")
        return []
    
    # Get query embedding
    query_embedding = get_query_embedding(query)
    if query_embedding is None:
        print("[ERROR] Could not get query embedding")
        return []
    
    try:
        # Build filter
        where_filter = None
        if hospital_names:
            if len(hospital_names) == 1:
                where_filter = {"hospital_name": hospital_names[0]}
            else:
                where_filter = {"hospital_name": {"$in": hospital_names}}
        
        # Query ChromaDB
        results = _chroma_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where_filter
        )
        
        # Parse results
        docs = []
        if results and results['documents']:
            # results is a dict of lists of lists (one list per query)
            flat_documents = results['documents'][0]
            flat_metadatas = results['metadatas'][0]
            
            for text, meta in zip(flat_documents, flat_metadatas):
                class Result:
                    def __init__(self, text, metadata):
                        self.page_content = text
                        self.metadata = metadata
                docs.append(Result(text, meta))
        
        print(f"[OK] Found {len(docs)} similar documents")
        return docs
        
    except Exception as e:
        print(f"[ERROR] ChromaDB search failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def index_hospital_reviews(hospitals: List[dict]) -> int:
    """Index hospital reviews into vector store"""
    
    texts = []
    metadatas = []
    
    # Extract texts from hospitals
    for h in hospitals:
        name = h.get("name", "Unknown")
        lat = h.get("lat")
        lon = h.get("lon")
        
        # Index aspect-based reviews
        for aspect_data in h.get("aspects", []):
            if len(aspect_data) >= 3:
                aspect, sentiment, line = aspect_data[0], aspect_data[1], aspect_data[2]
                if line and str(line).strip():
                    # Prefix with aspect info to make it retrievable
                    text = f"[{aspect} - {sentiment}]: {str(line).strip()}"
                    texts.append(text)
                    metadatas.append({
                        "hospital_name": name,
                        "aspect": aspect,
                        "sentiment": sentiment,
                        "lat": lat,
                        "lon": lon
                    })
        
        # Index general reviews
        for review in h.get("reviews", []):
            for sentence in _split_sentences(review):
                if sentence:
                    texts.append(sentence)
                    metadatas.append({
                        "hospital_name": name,
                        "aspect": "general",
                        "sentiment": "Unknown",
                        "lat": lat,
                        "lon": lon
                    })
        
        # Index aspect summaries if available
        if "aspect_summary" in h:
            for aspect, counts in h["aspect_summary"].items():
                pos = counts.get("Positive", 0)
                neg = counts.get("Negative", 0)
                summary_text = f"sentiment summary for {aspect} at {name}: {pos} positive mentions, {neg} negative mentions."
                texts.append(summary_text)
                metadatas.append({
                    "hospital_name": name,
                    "aspect": aspect,
                    "sentiment": "summary",
                    "lat": lat,
                    "lon": lon
                })
    
    if not texts:
        print("No texts to index")
        return 0
    
    print(f"Indexing {len(texts)} text chunks from hospitals...")
    added = add_documents(texts, metadatas)
    print(f"[OK] Successfully indexed {added} hospital documents")
    return added

def index_csv_reviews(reviews: List[str]) -> int:
    """Index CSV reviews into vector store"""
    
    texts = []
    metadatas = []
    
    # Extract sentences from reviews
    for review in reviews:
        for sentence in _split_sentences(review):
            if sentence:
                texts.append(sentence)
                metadatas.append({
                    "hospital_name": "CSV Upload",
                    "aspect": "general",
                    "sentiment": "Unknown"
                })
    
    if not texts:
        print("No texts to index from CSV")
        return 0
    
    print(f"Indexing {len(texts)} text chunks from CSV...")
    added = add_documents(texts, metadatas)
    print(f"[OK] Successfully indexed {added} CSV documents")
    return added

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class LocationRequest(BaseModel):
    location: Optional[str] = None
    use_auto: bool = False
    lat: Optional[float] = None
    lon: Optional[float] = None

class ReviewRequest(BaseModel):
    reviews: List[str]

class PDFRequest(BaseModel):
    aspect_summary: dict
    aspects: List[List[Any]]

class SentBotRequest(BaseModel):
    question: str
    hospital_names: Optional[List[str]] = None

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "api_key": GOOGLE_MAPS_API_KEY})

@app.post("/api/analyze-location")
async def analyze_location(request: LocationRequest):
    if request.use_auto and request.lat and request.lon:
        lat, lon = request.lat, request.lon
    elif request.location:
        lat, lon = get_coordinates(request.location)
        if not lat or not lon:
            raise HTTPException(status_code=400, detail="Invalid location")
    else:
        raise HTTPException(status_code=400, detail="Location required")

    try:
        hospitals = fetch_hospitals(lat, lon)[:10]
    except Exception:
        hospitals = fallback_hospitals(lat, lon)[:10]

    valid_hospitals = []
    for h in hospitals:
        aspects, sentiment_count = analyze_reviews(h["reviews"])
        h["positive_ratio"] = sentiment_count["Positive"] / (sentiment_count["Positive"] + sentiment_count["Negative"] + 1e-5)
        h["aspect_summary"] = {a: {"Positive": sum(1 for x, s, _ in aspects if x == a and s == "Positive"),
                              "Negative": sum(1 for x, s, _ in aspects if x == a and s == "Negative")}
                              for a in set(x for x, _, _ in aspects)}
        h["aspects"] = [[a, s, l] for a, s, l in aspects]

        text_blob = " ".join([line for _, _, line in aspects])
        if text_blob.strip():
            try:
                wc = generate_wordcloud(text_blob)
                img_buffer = io.BytesIO()
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                img_buffer.seek(0)
                h["wordcloud"] = base64.b64encode(img_buffer.read()).decode('utf-8')
            except Exception:
                pass

        valid_hospitals.append(h)

    top_hospitals = sorted(valid_hospitals, key=score_hospital, reverse=True)[:5]
    for h in top_hospitals:
        h["score"] = score_hospital(h)
    
    # Index hospitals for RAG
    try:
        count = index_hospital_reviews(valid_hospitals)
        print(f"[OK] Indexed {count} documents from {len(valid_hospitals)} hospitals")
    except Exception as e:
        print(f"[ERROR] Failed to index hospitals: {e}")
    
    return JSONResponse(content={
        "hospitals": valid_hospitals,
        "top_hospitals": top_hospitals
    })

@app.post("/api/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if "review" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'review' column")
        
        reviews_list = df['review'].dropna().tolist()
        aspects, sentiment_count = analyze_reviews(reviews_list, include_general=True)
        
        aspect_summary = {a: {"Positive": sum(1 for x, s, _ in aspects if x == a and s == "Positive"), 
                             "Negative": sum(1 for x, s, _ in aspects if x == a and s == "Negative")} 
                         for a in set(x for x, _, _ in aspects)}
        
        text_blob = " ".join([line for _, _, line in aspects])
        wordcloud_b64 = None
        if text_blob.strip():
            wc = generate_wordcloud(text_blob)
            img_buffer = io.BytesIO()
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            img_buffer.seek(0)
            wordcloud_b64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        
        # Index CSV reviews for RAG
        try:
            count = index_csv_reviews(reviews_list)
            print(f"[OK] Indexed {count} documents from CSV")
        except Exception as e:
            print(f"[ERROR] Failed to index CSV reviews: {e}")
        
        pie_data = Counter([a for a, _, _ in aspects])
        pie_chart_data = [{"aspect": k, "count": v} for k, v in pie_data.items()]
        
        return JSONResponse(content={
            "aspect_summary": aspect_summary,
            "aspects": [[a, s, l] for a, s, l in aspects],
            "sentiment_count": sentiment_count,
            "wordcloud": wordcloud_b64,
            "pie_chart": pie_chart_data
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-pdf")
async def generate_pdf_endpoint(request: PDFRequest):
    try:
        from visualizations import generate_pdf_report
        aspects = [tuple(a) for a in request.aspects]
        pdf_bytes = generate_pdf_report(request.aspect_summary, aspects)
        return JSONResponse(content={"pdf": base64.b64encode(pdf_bytes).decode('utf-8')})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RAG / SENTBOT ENDPOINTS
# ============================================================================

@app.get("/api/rag-stats")
async def rag_stats():
    """Get RAG statistics"""
    count = 0
    if _chroma_collection:
        count = _chroma_collection.count()
    
    return JSONResponse(content={
        "count": count,
        "cache_size": len(_embeddings_cache),
        "status": "active" if count > 0 else "empty"
    })

@app.post("/api/rag-add-samples")
async def rag_add_samples():
    """Add sample data to RAG for testing"""
    try:
        samples = [
            "The hospital has clean wards and friendly staff.",
            "Queue at reception was long but nurses were helpful.",
            "Rooms were clean and sanitized; staff behavior was polite.",
            "Doctors were very professional and caring.",
            "Emergency services were quick and efficient."
        ]
        
        count = index_csv_reviews(samples)
        
        return JSONResponse(content={
            "ok": count > 0,
            "message": f"Added {count} sample documents"
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})

@app.get("/api/rag-indexed-count")
async def rag_indexed_count():
    """Get count of indexed documents"""
    count = 0
    if _chroma_collection:
        count = _chroma_collection.count()
        
    return JSONResponse(content={
        "count": count,
        "system": "chromadb"
    })

@app.get("/api/rag-debug")
async def rag_debug():
    """Debug RAG system"""
    # Test embedding generation
    test_embedding = None
    test_error = None
    try:
        _configure_groq()
        test_embedding = get_embedding("test hospital review")
        if test_embedding is not None:
            test_embedding = f"Success: shape={test_embedding.shape}, dtype={test_embedding.dtype}"
    except Exception as e:
        test_error = str(e)
        import traceback
        test_error += "\n" + traceback.format_exc()
    
    # Sample document info
    count = 0
    sample_docs = []
    if _chroma_collection:
        count = _chroma_collection.count()
        if count > 0:
            res = _chroma_collection.peek(limit=3)
            # res is dict of lists
            ids = res['ids']
            docs = res['documents']
            for i, doc_id in enumerate(ids):
                sample_docs.append({
                    "id": doc_id,
                    "text_preview": docs[i][:50] + "...",
                    "has_embedding": True
                })
    
    return JSONResponse(content={
        "system": "chromadb",
        "document_count": count,
        "cache_size": len(_embeddings_cache),
        "groq_configured": bool(GROQ_API_KEY),
        "groq_key_present": bool(GROQ_API_KEY),
        "chroma_db_path": os.path.join(os.getcwd(), "chroma_db"),
        "test_embedding": test_embedding,
        "test_error": test_error,
        "sample_documents": sample_docs
    })

@app.post("/api/rag-reset")
async def rag_reset():
    """Reset and reinitialize the RAG system"""
    global _chroma_client, _chroma_collection
    
    try:
        print("\n=== Resetting RAG System ===")
        
        if _chroma_client:
            try:
                _chroma_client.delete_collection("hospital_reviews")
                print("[OK] Deleted collection")
            except Exception as e:
                print(f"[WARNING] Could not delete collection: {e}")
            
            # Recreate
            try:
                _chroma_collection = _chroma_client.get_or_create_collection(
                    name="hospital_reviews",
                    metadata={"hnsw:space": "cosine"}
                )
                print("[OK] Recreated collection")
            except Exception as e:
                print(f"[ERROR] Could not recreate collection: {e}")
        
        _embeddings_cache = {}
        
        return JSONResponse(content={
            "success": True,
            "message": "RAG system reset successfully"
        })
        
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        })

@app.post("/api/sentbot-ask")
async def sentbot_ask(req: SentBotRequest):
    """SentBot RAG endpoint"""
    
    try:
        # Ensure Groq is configured
        if not GROQ_API_KEY:
            return JSONResponse(content={
                "answer": "Groq API key is not configured. Please set GROQ_API_KEY environment variable.",
                "sources": []
            })
        
        _configure_groq()
        
        # Retrieve relevant documents
        # Increase k to 20 to ensure we get both specific reviews AND aspect summaries
        docs = search_documents(
            req.question, 
            k=20, 
            hospital_names=req.hospital_names
        )
        
        # If no documents found, return default message
        if not docs:
            return JSONResponse(content={
                "answer": "I don't have enough information to answer that question. Please analyze some hospitals or upload CSV data first to populate the knowledge base.",
                "sources": []
            })
        
        # Build context from retrieved documents
        # OPTIMIZATION: Use more documents (k=20) but keep truncation reasonable
        context_parts = []
        for doc in docs[:20]:
            try:
                hospital_name = doc.metadata.get("hospital_name", "Unknown")
                aspect = doc.metadata.get("aspect", "general")
                content = doc.page_content
                # Truncate content to max 500 chars per doc
                if content:
                    content = content[:500] + "..." if len(content) > 500 else content 
                    context_parts.append(f"[{hospital_name} - {aspect}]: {content}")
            except Exception as e:
                print(f"[WARNING] Error processing document: {e}")
                continue
        
        if not context_parts:
            return JSONResponse(content={
                "answer": "I found documents but couldn't extract the content. Please try again.",
                "sources": []
            })
        
        context = "\n".join(context_parts)
        
        # Create prompt
        system_prompt = """You are SentBot, a professional AI assistant dedicated to helping users understand hospital reviews and sentiment. 
Your goal is to provide precise, clear, and helpful answers based on the provided hospital review data.
Use your general medical and world knowledge to enhance your answers, but strictly ground factual claims about specific hospitals in the provided context."""

        user_prompt = f"""Task Instructions:
1. **Analyze Context**: Use the provided hospital reviews (organized by aspect) to answer the user's question.
2. **Use Intelligence**: Combine the specific review details with your own general knowledge to construct a well-reasoned and natural response.
3. **No Unsolicited Comparisons**: Do NOT compare hospitals unless the user explicitly asks for a comparison. Focus only on the hospital(s) relevant to the query.
4. **Style & Tone**: Keep the answer precise, clear, and simple. Maintain a professional and helpful tone.
5. **Handling Limitations**: If the context does not contain the answer, politely state that the information is not available in the current reviews.

Context (Hospital Reviews):
{context}

Question: {req.question}

Answer:"""
        
        # Generate answer using Groq
        answer_text = None
        
        try:
            if _groq_client:
                chat_completion = _groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    max_tokens=512,
                )
                answer_text = chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer with Groq: {e}")
            
        
        if not answer_text:
            # =========================================================================
            # FALLBACK MODE (MANUAL ANSWER)
            # =========================================================================
            print("[INFO] Using Fallback Mode (No API)")
            
            top_doc = docs[0]
            h_name = top_doc.metadata.get("hospital_name", "the hospital")
            content = top_doc.page_content
            
            answer_text = (
                f"I am SentBot, your assistant for hospital reviews. "
                f"I am currently experiencing high traffic and cannot generate a custom comprehensive answer right now. "
                f"However, I found a relevant review for {h_name} that might help:\n\n"
                f"\"{content}\"\n\n"
                f"Please ask again in a moment for a full analysis."
            )
        
        # Prepare sources
        sources = []
        for doc in docs[:5]:
            try:
                sources.append({
                    "hospital_name": doc.metadata.get("hospital_name", "Unknown"),
                    "aspect": doc.metadata.get("aspect", "general"),
                    "sentiment": doc.metadata.get("sentiment", "Unknown"),
                    "text": doc.page_content[:200] if doc.page_content else ""
                })
            except Exception as e:
                print(f"[WARNING] Error preparing source: {e}")
                continue
        
        return JSONResponse(content={
            "answer": answer_text,
            "sources": sources
        })
    
    except Exception as e:
        print(f"Error in sentbot_ask: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={
            "answer": f"An error occurred: {str(e)}",
            "sources": []
        })

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG on startup"""
    print("\n" + "="*60)
    print("Starting Hospital Review Analysis System")
    print("="*60)
    
    # Initialize vectorstore
    _init_vectorstore()
    
    print("\nSystem ready!")
    print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)