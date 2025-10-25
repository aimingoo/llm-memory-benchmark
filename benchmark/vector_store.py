"""
Simple vector store abstraction using Chroma (or FAISS stub).
This is intentionally small — replace/extend for production use.
"""
import os
from typing import List, Dict

def get_env(key, default=None):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return os.environ.get(key, default)

class VectorStoreWrapper:
    def __init__(self, backend="chroma", collection_name="memory_bench"):
        self.backend = backend
        if backend == "chroma":
            import chromadb
            from chromadb.config import Settings
            persist_dir = get_env("CHROMA_DIR", "./chroma_db")
            self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
            # create/get collection
            try:
                # delete collection if exists for clean runs
                self.client.delete_collection(name=collection_name)
            except Exception:
                pass
            self.col = self.client.create_collection(collection_name)
        else:
            # FAISS basic in-memory placeholder using lists (not efficient; replace for production)
            self.doc_texts = []
            self.metadatas = []
            self.ids = []

    def add(self, ids: List[str], texts: List[str], metadatas: List[Dict]):
        if self.backend == "chroma":
            self.col.add(documents=texts, metadatas=metadatas, ids=ids)
            self.client.persist()
        else:
            self.ids.extend(ids)
            self.doc_texts.extend(texts)
            self.metadatas.extend(metadatas)

    def query(self, q: str, top_k=5):
        if self.backend == "chroma":
            results = self.col.query(query_texts=[q], n_results=top_k)
            # results: dict with 'ids','documents','metadatas','distances'
            return [{"id": i, "text": t, "metadata": m} for i,t,m in zip(results["ids"][0], results["documents"][0], results["metadatas"][0])]
        else:
            # naive text-sim by substring score (fallback)
            scored = []
            for idx, text in enumerate(self.doc_texts):
                score = 1.0 if q in text else 0.0
                scored.append((score, self.ids[idx], text, self.metadatas[idx]))
            scored.sort(reverse=True)
            return [{"id": sid, "text": txt, "metadata": md} for _, sid, txt, md in scored[:top_k]]