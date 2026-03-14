import os
import pickle
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()

ROOT       = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "chroma"
BM25_DIR   = ROOT / "data" / "bm25"

EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME  = "fda_labels"
NUM_QUERIES      = 3     # query variants to generate
DENSE_TOP_K      = 20    # candidates from dense search
SPARSE_TOP_K     = 20    # candidates from sparse search
RERANK_TOP_K     = 5     # final chunks after reranking
RRF_K            = 60    # RRF constant (standard default)


class RetrievalState(TypedDict):
    original_query:    str
    reformulated:      list[str]
    dense_results:     list[dict]
    sparse_results:    list[dict]
    fused_results:     list[dict]
    reranked_results:  list[dict]



class HybridRetriever:
    def __init__(self):
        print("[retriever] Loading models and indexes...")
        self.embed_model   = SentenceTransformer(EMBED_MODEL)
        self.rerank_model  = CrossEncoder(RERANK_MODEL)
        self.llm           = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.collection    = self._load_chroma()
        self.bm25, self.corpus_texts, self.corpus_metadatas = self._load_bm25()
        self.graph         = self._build_graph()
        print("[retriever] Ready.")

    def _load_chroma(self):
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        return client.get_collection(COLLECTION_NAME)

    def _load_bm25(self) -> tuple[BM25Okapi, list[str], list[dict]]:
        with open(BM25_DIR / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        with open(BM25_DIR / "corpus.pkl", "rb") as f:
            corpus = pickle.load(f)
        return bm25, corpus["texts"], corpus["metadatas"]

    def _node_reformulate(self, state: RetrievalState) -> RetrievalState:
        """
        Generate NUM_QUERIES alternative phrasings of the original query.
        This improves recall by catching cases where the user's phrasing
        doesn't match the terminology used in FDA label sections.
        """
        prompt = f"""You are helping retrieve information from FDA drug labels.
Given the user query below, generate {NUM_QUERIES} alternative phrasings that:
- Use clinical/medical terminology where appropriate
- Vary the structure (question form, keyword form, etc.)
- Preserve the original intent exactly

Return ONLY the {NUM_QUERIES} alternatives, one per line, no numbering or explanation.

User query: {state['original_query']}"""

        response = self.llm.invoke(prompt)
        variants = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip()
        ][:NUM_QUERIES]

        # include original with the variants for comparison
        all_queries = [state["original_query"]] + variants

        return {**state, "reformulated": all_queries}

    def _node_dense_retrieve(self, state: RetrievalState) -> RetrievalState:
        #embed all query variants
        seen_ids = set()
        results  = []

        for query in state["reformulated"]:
            embedding = self.embed_model.encode(
                query, normalize_embeddings=True
            ).tolist()

            response = self.collection.query(
                query_embeddings=[embedding],
                n_results=DENSE_TOP_K,
                include=["documents", "metadatas", "distances"],
            )

            for doc, meta, dist in zip(
                response["documents"][0],
                response["metadatas"][0],
                response["distances"][0],
            ):
                chunk_id = f"{meta['source_id']}_{meta['section']}_{meta['chunk_index']}"
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    results.append({
                        "text":      doc,
                        "metadata":  meta,
                        "chunk_id":  chunk_id,
                        "score":     1 - dist,  # cosine distance → similarity
                    })

        return {**state, "dense_results": results}

    def _node_sparse_retrieve(self, state: RetrievalState) -> RetrievalState:
        #bm25 keyword search using original query
        tokenized_query = state["original_query"].lower().split()
        scores          = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:SPARSE_TOP_K]

        results = []
        for idx in top_indices:
            meta     = self.corpus_metadatas[idx]
            chunk_id = f"{meta['source_id']}_{meta['section']}_{meta['chunk_index']}"
            results.append({
                "text":     self.corpus_texts[idx],
                "metadata": meta,
                "chunk_id": chunk_id,
                "score":    float(scores[idx]),
            })

        return {**state, "sparse_results": results}

    def _node_fuse(self, state: RetrievalState) -> RetrievalState:
        #combine both dense and sparse ranked lists
        rrf_scores: dict[str, float] = {}
        chunk_map:  dict[str, dict]  = {}

        for rank, item in enumerate(state["dense_results"]):
            cid = item["chunk_id"]
            rrf_scores[cid]  = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
            chunk_map[cid]   = item

        for rank, item in enumerate(state["sparse_results"]):
            cid = item["chunk_id"]
            rrf_scores[cid]  = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
            chunk_map[cid]   = item

        fused = sorted(chunk_map.values(), key=lambda x: rrf_scores[x["chunk_id"]], reverse=True)

        # Attach RRF score for transparency
        for item in fused:
            item["rrf_score"] = rrf_scores[item["chunk_id"]]

        return {**state, "fused_results": fused}

    def _node_rerank(self, state: RetrievalState) -> RetrievalState:
        #score candidates against original query, can replace with simple cosine sim if too slow
        query     = state["original_query"]
        candidates = state["fused_results"]

        if not candidates:
            return {**state, "reranked_results": []}

        pairs  = [(query, item["text"]) for item in candidates]
        scores = self.rerank_model.predict(pairs)

        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return {**state, "reranked_results": reranked[:RERANK_TOP_K]}

    def _build_graph(self):
        graph = StateGraph(RetrievalState)

        graph.add_node("reformulate",     self._node_reformulate)
        graph.add_node("dense_retrieve",  self._node_dense_retrieve)
        graph.add_node("sparse_retrieve", self._node_sparse_retrieve)
        graph.add_node("fuse",            self._node_fuse)
        graph.add_node("rerank",          self._node_rerank)

        graph.set_entry_point("reformulate")
        graph.add_edge("reformulate",     "dense_retrieve")
        graph.add_edge("dense_retrieve",  "sparse_retrieve")
        graph.add_edge("sparse_retrieve", "fuse")
        graph.add_edge("fuse",            "rerank")
        graph.add_edge("rerank",          END)

        return graph.compile()

    def retrieve(self, query: str) -> dict:
        #runing full retrieval pipelien for queries and return state dict
        initial_state: RetrievalState = {
            "original_query":   query,
            "reformulated":     [],
            "dense_results":    [],
            "sparse_results":   [],
            "fused_results":    [],
            "reranked_results": [],
        }

        return self.graph.invoke(initial_state)


def build_retriever() -> HybridRetriever:
    return HybridRetriever()

if __name__ == "__main__":
    retriever = build_retriever()

    test_query = "What are the contraindications for metformin in patients with kidney disease?"
    print(f"\nQuery: {test_query}\n")

    state = retriever.retrieve(test_query)

    print("Reformulated queries:")
    for q in state["reformulated"]:
        print(f"  - {q}")

    print(f"\nTop {RERANK_TOP_K} chunks after reranking:")
    for i, chunk in enumerate(state["reranked_results"], 1):
        meta = chunk["metadata"]
        print(f"\n[{i}] {meta['drug_name']} — {meta['section']}")
        print(f"     Rerank score: {chunk['rerank_score']:.4f}")
        print(f"     RRF score:    {chunk['rrf_score']:.4f}")
        print(f"     Text: {chunk['text'][:200]}...")