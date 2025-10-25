"""
Core benchmark logic:
- For each test scenario: simulate incremental events (each event may append memory).
- At specified query points, perform retrieval (top-k) and call LLM for answer.
- Compute simple metrics: exact_match, contains_expected, retrieved_topk_recall.
"""
import os
import time
import uuid
import openai
from typing import List, Dict
from .vector_store import VectorStoreWrapper
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

class MemoryBenchmark:
    def __init__(self, vectorstore="chroma", embedding_model="text-embedding-3-small", llm_model="gpt-4o-mini"):
        self.vs = VectorStoreWrapper(backend=vectorstore)
        self.embedding_model = embedding_model
        self.llm_model = llm_model

    def _embed(self, texts: List[str]):
        # Use OpenAI embeddings API (batch)
        if not OPENAI_API_KEY:
            # fallback: return dummy zero vectors (vectorstore may use text-only fallback)
            return [[0.0]]*len(texts)
        resp = openai.Embedding.create(model=self.embedding_model, input=texts)
        return [d["embedding"] for d in resp["data"]]

    def _llm_generate(self, prompt: str, temperature=0.0):
        if not OPENAI_API_KEY:
            return "DUMMY_RESPONSE_NO_API_KEY"
        resp = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=300
        )
        return resp["choices"][0]["message"]["content"].strip()

    def _insert_memory(self, text: str, metadata: Dict):
        mid = metadata.get("id", str(uuid.uuid4()))
        # store doc into vectorstore (Chroma handles embeddings internally if configured, but we keep simple)
        self.vs.add([mid], [text], [metadata])
        return mid

    def run_single_scenario(self, scenario: Dict):
        """
        scenario structure:
        {
          "id": "case1",
          "description": "persona retention",
          "events": [
             {"type":"utterance","text":"My name is Alice.","store_memory": true, "memory_id":"m1"},
             ...
          ],
          "queries": [
             {"at_event": 2, "query":"What is the user's name?", "expected":"Alice", "top_k":5}
          ]
        }
        """
        # reset vector store per scenario by re-init
        self.vs = VectorStoreWrapper(backend=self.vs.backend)
        results = {"scenario_id": scenario.get("id"), "description": scenario.get("description"), "tests": []}
        events = scenario.get("events", [])
        queries = scenario.get("queries", [])
        # process events incrementally
        for i, ev in enumerate(events, start=1):
            if ev.get("store_memory"):
                text = ev.get("text")
                metadata = ev.get("metadata", {})
                metadata.update({"event_index": i})
                self._insert_memory(text, metadata)
            # check queries that trigger at this event
            for q in [qq for qq in queries if qq.get("at_event")==i]:
                # retrieval
                retrieved = self.vs.query(q["query"], top_k=q.get("top_k",5))
                ctx = "\n\n".join([r["text"] for r in retrieved])
                prompt = f"Context:\n{ctx}\n\nQuestion: {q['query']}\nAnswer briefly."
                gen = self._llm_generate(prompt, temperature=0.0)
                # metrics
                expected = q.get("expected","").strip()
                exact = (gen.strip()==expected) if expected else False
                contains = (expected in gen) if expected else False
                # topk recall: if expected memory id provided, check if retrieved contains it
                expected_mem_id = q.get("expected_memory_id")
                retrieved_ids = [r.get("id") for r in retrieved]
                topk_recall = (expected_mem_id in retrieved_ids) if expected_mem_id else None
                results["tests"].append({
                    "at_event": i,
                    "query": q["query"],
                    "generated": gen,
                    "expected": expected,
                    "exact_match": exact,
                    "contains_expected": contains,
                    "topk_recall": topk_recall,
                    "retrieved_ids": retrieved_ids
                })
        return results

    def run_tests(self, tests: List[Dict]):
        all_results = {"meta": {"time": time.time()}, "scenarios": []}
        for sc in tests:
            res = self.run_single_scenario(sc)
            all_results["scenarios"].append(res)
        return all_results