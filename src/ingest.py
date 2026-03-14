import os
import json
import pickle
import time
import re
import requests

from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

load_dotenv()

# Paths
ROOT        = Path(__file__).resolve().parent.parent
RAW_DIR     = ROOT / "data" / "raw"
CHROMA_DIR  = ROOT / "data" / "chroma"
BM25_DIR    = ROOT / "data" / "bm25"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
BM25_DIR.mkdir(parents=True, exist_ok=True)

# Config stuff
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "fda_labels"
BATCH_SIZE      = 100       # openFDA max per request
NUM_BATCHES     = 10        # 10 x 100 = 1,000 labels
MAX_CHUNK_CHARS = 800       # soft ceiling per chunk
CHUNK_OVERLAP   = 80        # overlap between chunks in same section

SECTIONS = [
    "boxed_warning",
    "indications_and_usage",
    "contraindications",
    "warnings_and_cautions",
    "warnings",
    "adverse_reactions",
    "drug_interactions",
    "dosage_and_administration",
    "overdosage",
    "mechanism_of_action",
    "clinical_pharmacology",
]

FDA_API = "https://api.fda.gov/drug/label.json"

def fetch_labels(num_batches: int = NUM_BATCHES) -> list[dict]:
    #pull labels from api and cache locally
    cache_path = RAW_DIR / "labels_raw.json"

    if cache_path.exists():
        print(f"[ingest] Found cached labels at {cache_path} — skipping fetch.")
        with open(cache_path) as f:
            return json.load(f)

    print(f"[ingest] Fetching {num_batches * BATCH_SIZE} labels from openFDA...")
    all_labels = []

    for i in tqdm(range(num_batches), desc="Fetching batches"):
        params = {
            "limit": BATCH_SIZE,
            "skip":  i * BATCH_SIZE,
        }
        try:
            resp = requests.get(FDA_API, params=params, timeout=30)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            all_labels.extend(results)
            time.sleep(0.25)  # be polite to the API
        except requests.RequestException as e:
            print(f"[ingest] Warning: batch {i} failed ({e}) — continuing.")
            continue

    # Filter to labels that have at least one useful section
    filtered = [
        label for label in all_labels
        if any(label.get(s) for s in SECTIONS)
    ]

    print(f"[ingest] Fetched {len(all_labels)} labels, {len(filtered)} with usable sections.")

    with open(cache_path, "w") as f:
        json.dump(filtered, f)

    return filtered


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_section(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    #split sections on sentences
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from end of previous
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = (overlap_text + " " + sentence).strip()

    if current:
        chunks.append(current)

    return chunks


def extract_drug_name(label: dict) -> str:
    """Pull the most useful name from a label record."""
    openfda = label.get("openfda", {})
    for field in ("brand_name", "generic_name", "substance_name"):
        names = openfda.get(field, [])
        if names:
            return names[0].title()
    return "Unknown Drug"


def chunk_labels(labels: list[dict]) -> tuple[list[str], list[dict]]:
    #just converting labels into tuples to pair text to extracted data
    texts, metadatas = [], []

    for label in tqdm(labels, desc="Chunking labels"):
        drug_name = extract_drug_name(label)
        source_id = label.get("id", "unknown")

        for section in SECTIONS:
            raw = label.get(section)
            if not raw:
                continue

            section_text = clean_text(" ".join(raw) if isinstance(raw, list) else raw)

            if len(section_text) < 30:  #throw out short sections
                continue

            chunks = split_section(section_text)

            for idx, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                    "drug_name":   drug_name,
                    "section":     section,
                    "chunk_index": idx,
                    "source_id":   source_id,
                    "num_chunks":  len(chunks),
                })

    print(f"[ingest] Produced {len(texts)} chunks from {len(labels)} labels.")
    return texts, metadatas

def build_chroma_index(texts: list[str], metadatas: list[dict]) -> None:
    #make a chromadb collection for cache
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"[ingest] Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    embed_batch = 256
    all_embeddings = []

    for i in tqdm(range(0, len(texts), embed_batch), desc="Embedding"):
        batch = texts[i : i + embed_batch]
        embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.extend(embeddings.tolist())

    store_batch = 5000
    ids = [f"chunk_{i}" for i in range(len(texts))]

    for i in tqdm(range(0, len(texts), store_batch), desc="Storing in ChromaDB"):
        collection.add(
            ids=ids[i : i + store_batch],
            documents=texts[i : i + store_batch],
            embeddings=all_embeddings[i : i + store_batch],
            metadatas=metadatas[i : i + store_batch],
        )

    print(f"[ingest] ChromaDB collection '{COLLECTION_NAME}' built with {len(texts)} chunks.")


def build_bm25_index(texts: list[str], metadatas: list[dict]) -> None:
    #for ranking responses later
    print("[ingest] Building BM25 index...")

    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)

    with open(BM25_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(BM25_DIR / "corpus.pkl", "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)

    print(f"[ingest] BM25 index saved to {BM25_DIR}/")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FDA RAG — Ingestion Pipeline")
    print("=" * 60)

    labels   = fetch_labels()
    texts, metadatas = chunk_labels(labels)
    build_chroma_index(texts, metadatas)
    build_bm25_index(texts, metadatas)

    print("\n[ingest] Done. Corpus ready for retrieval.")
    print(f"  Chunks total : {len(texts)}")
    print(f"  ChromaDB     : {CHROMA_DIR}")
    print(f"  BM25 index   : {BM25_DIR}")


if __name__ == "__main__":
    main()
