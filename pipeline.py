# pip install requests beautifulsoup4 sentence-transformers chromadb rank_bm25 pdfplumber

import json
import re
import time
import pdfplumber
import requests
import chromadb

from bs4 import BeautifulSoup
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ─── globals ──────────────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
tokenizer = model.tokenizer

chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection("ualbany_informatics")

# BM25 index is rebuilt each time embed_and_store is called
_bm25_index = None
_bm25_chunks = []

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0; contact: youremail@gmail.com)",
    "Accept": "application/json",
}


# ─── source metadata defaults ─────────────────────────────────────────────────

SOURCE_DEFAULTS = {
    "reddit": {
        "source": "reddit",
        "source_authority": 0.5,
    },
    "coursicle": {
        "source": "coursicle",
        "source_authority": 0.6,
    },
    "ratemyprofessor": {
        "source": "ratemyprofessor",
        "source_authority": 0.7,
    },
    "collegeconfidential": {
        "source": "collegeconfidential",
        "source_authority": 0.5,
    },
    "niche": {
        "source": "niche",
        "source_authority": 0.6,
    },
    "manual": {
        "source": "manual",
        "source_authority": 0.5,
    },
    "pdf": {
        "source": "pdf",
        "source_authority": 0.8,
    },
}


# ─── helpers ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _decode(token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─── load_source ──────────────────────────────────────────────────────────────

def load_source(url_or_path: str, source_type: str) -> str:
    """
    Fetch and return raw text from a source.

    source_type options:
        "reddit"             — uses the .json API endpoint
        "coursicle"          — HTML scrape, extracts review text
        "ratemyprofessor"    — HTML scrape; returns MANUAL_LOAD_REQUIRED if blocked
        "collegeconfidential"— HTML scrape, extracts forum posts
        "niche"              — HTML scrape, extracts review cards
        "manual"             — reads a local .txt file from disk
    """

    source_type = source_type.lower().strip()

    # ── reddit ────────────────────────────────────────────────────────────────
    if source_type == "reddit":
        json_url = url_or_path.rstrip("/") + ".json?limit=500"
        resp = requests.get(json_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        texts = []

        def extract_comments(node):
            if not isinstance(node, dict):
                return
            kind = node.get("kind")
            d = node.get("data", {})
            if kind == "t3":
                title = d.get("title", "")
                body = d.get("selftext", "")
                if title:
                    texts.append(f"POST TITLE: {title}")
                if body and body not in ("[removed]", "[deleted]"):
                    texts.append(body)
            elif kind == "t1":
                body = d.get("body", "")
                if body and body not in ("[removed]", "[deleted]"):
                    texts.append(body)
            for reply in (
                d.get("replies", {}).get("data", {}).get("children", [])
                if isinstance(d.get("replies"), dict)
                else []
            ):
                extract_comments(reply)
            for child in d.get("children", []):
                extract_comments(child)

        for listing in data:
            extract_comments(listing)

        return _clean_text("\n\n".join(texts))

    # ── coursicle ─────────────────────────────────────────────────────────────
    elif source_type == "coursicle":
        resp = requests.get(url_or_path, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        reviews = []
        for el in soup.select(".review, .reviewText, [class*='review']"):
            t = el.get_text(separator=" ", strip=True)
            if len(t) > 30:
                reviews.append(t)

        if not reviews:
            for p in soup.find_all("p"):
                t = p.get_text(strip=True)
                if len(t) > 40:
                    reviews.append(t)

        return _clean_text("\n\n".join(reviews)) if reviews else "MANUAL_LOAD_REQUIRED"

    # ── ratemyprofessor ───────────────────────────────────────────────────────
    elif source_type == "ratemyprofessor":
        try:
            resp = requests.get(url_or_path, headers=HEADERS, timeout=15)
            if resp.status_code in (403, 429, 503):
                print(f"[ratemyprofessor] Blocked ({resp.status_code}). "
                      "Save page text manually to a .txt file and use source_type='manual'.")
                return "MANUAL_LOAD_REQUIRED"
            soup = BeautifulSoup(resp.text, "html.parser")
            reviews = []
            for el in soup.select("[class*='Comments__StyledComments'], [class*='rating-list']"):
                t = el.get_text(separator=" ", strip=True)
                if len(t) > 30:
                    reviews.append(t)
            if not reviews:
                print("[ratemyprofessor] No reviews parsed — likely JS-rendered. "
                      "Use source_type='manual' with a saved .txt file.")
                return "MANUAL_LOAD_REQUIRED"
            return _clean_text("\n\n".join(reviews))
        except Exception as e:
            print(f"[ratemyprofessor] Error: {e}. Use source_type='manual'.")
            return "MANUAL_LOAD_REQUIRED"

    # ── college confidential ──────────────────────────────────────────────────
    elif source_type == "collegeconfidential":
        resp = requests.get(url_or_path, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        posts = []
        for el in soup.select(".post, .cooked, article, [class*='post-body']"):
            t = el.get_text(separator=" ", strip=True)
            if len(t) > 40:
                posts.append(t)

        return _clean_text("\n\n".join(posts)) if posts else "MANUAL_LOAD_REQUIRED"

    # ── niche ─────────────────────────────────────────────────────────────────
    elif source_type == "niche":
        resp = requests.get(url_or_path, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        reviews = []
        for el in soup.select("[class*='review__content'], [class*='ReviewCard'], .comment__content"):
            t = el.get_text(separator=" ", strip=True)
            if len(t) > 40:
                reviews.append(t)

        return _clean_text("\n\n".join(reviews)) if reviews else "MANUAL_LOAD_REQUIRED"

    # ── manual .txt file ──────────────────────────────────────────────────────
    elif source_type == "manual":
        with open(url_or_path, "r", encoding="utf-8") as f:
            return _clean_text(f.read())

    # ── local pdf ─────────────────────────────────────────────────────────────
    elif source_type == "pdf":
        pages = []
        with pdfplumber.open(url_or_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return _clean_text("\n\n".join(pages))

    else:
        raise ValueError(f"Unknown source_type: '{source_type}'. "
                         "Choose from: reddit, coursicle, ratemyprofessor, "
                         "collegeconfidential, niche, manual, pdf")


# ─── chunk_text ───────────────────────────────────────────────────────────────

def chunk_text(
    raw_text: str,
    source_meta: dict,
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[dict]:
    """
    Split raw_text into chunks of chunk_size tokens (max 250) with overlap.
    Respects double-newline boundaries (review/comment boundaries) first.
    Returns list of dicts: {text, source, url, topic_tag, date,
                             author_context, source_authority}
    """
    if raw_text == "MANUAL_LOAD_REQUIRED":
        print(f"[chunk_text] Skipping — source requires manual load: {source_meta.get('url')}")
        return []

    MAX_TOKENS = 250
    chunk_size = min(chunk_size, MAX_TOKENS)

    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

    token_buffer: list[int] = []
    overlap_tokens: list[int] = []
    chunks: list[dict] = []

    def flush(token_buffer):
        if not token_buffer:
            return
        text = _decode(token_buffer)
        if len(text.strip()) < 25:
            return
        chunk = {"text": text.strip()}
        chunk.update(source_meta)
        chunks.append(chunk)

    for para in paragraphs:
        para_tokens = _tokenize(para)

        if not para_tokens:
            continue

        # if paragraph fits in remaining buffer space, add it
        if len(token_buffer) + len(para_tokens) <= chunk_size:
            token_buffer.extend(para_tokens)
        else:
            # flush current buffer if it has content
            if token_buffer:
                flush(token_buffer)
                overlap_tokens = token_buffer[-overlap:]
                token_buffer = list(overlap_tokens)

            # if the paragraph itself exceeds chunk_size, split it
            if len(para_tokens) > chunk_size:
                pos = 0
                while pos < len(para_tokens):
                    available = chunk_size - len(token_buffer)
                    segment = para_tokens[pos: pos + available]
                    token_buffer.extend(segment)
                    if len(token_buffer) >= chunk_size:
                        flush(token_buffer)
                        overlap_tokens = token_buffer[-overlap:]
                        token_buffer = list(overlap_tokens)
                    pos += available
            else:
                token_buffer.extend(para_tokens)

    flush(token_buffer)
    return chunks


# ─── embed_and_store ──────────────────────────────────────────────────────────

def embed_and_store(chunks: list[dict]) -> None:
    """
    Encode each chunk with all-MiniLM-L6-v2 and upsert into ChromaDB.
    Also rebuilds the BM25 index over the full corpus.
    """
    global _bm25_index, _bm25_chunks

    if not chunks:
        print("[embed_and_store] No chunks to store.")
        return

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {k: v for k, v in c.items() if k != "text"}
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    # rebuild BM25
    tokenized = [t.lower().split() for t in texts]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_chunks = chunks

    print(f"[embed_and_store] Stored {len(chunks)} chunks.")


# ─── retrieve ─────────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = 6) -> list[dict]:
    """
    Hybrid retrieval: dense (ChromaDB cosine) + sparse (BM25), fused with RRF.
    Returns top-k chunks with fused_score added.
    """
    if _bm25_index is None:
        raise RuntimeError("Call embed_and_store() before retrieve().")

    RRF_K = 60
    n_candidates = min(len(_bm25_chunks), max(k * 4, 20))

    # ── dense retrieval ───────────────────────────────────────────────────────
    query_embedding = model.encode([query]).tolist()
    dense_results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"],
    )

    dense_ids = dense_results["ids"][0]
    dense_scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        dense_scores[doc_id] = 1.0 / (RRF_K + rank + 1)

    # ── sparse retrieval (BM25) ───────────────────────────────────────────────
    tokenized_query = query.lower().split()
    bm25_scores = _bm25_index.get_scores(tokenized_query)
    bm25_ranked = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:n_candidates]

    sparse_scores: dict[str, float] = {}
    for rank, idx in enumerate(bm25_ranked):
        doc_id = f"chunk_{idx}"
        sparse_scores[doc_id] = 1.0 / (RRF_K + rank + 1)

    # ── RRF fusion ────────────────────────────────────────────────────────────
    all_ids = set(dense_scores) | set(sparse_scores)
    fused: dict[str, float] = {
        doc_id: dense_scores.get(doc_id, 0.0) + sparse_scores.get(doc_id, 0.0)
        for doc_id in all_ids
    }
    top_ids = sorted(fused, key=lambda x: fused[x], reverse=True)[:k]

    # ── build result list ─────────────────────────────────────────────────────
    results = []
    for doc_id in top_ids:
        idx = int(doc_id.replace("chunk_", ""))
        chunk = dict(_bm25_chunks[idx])
        chunk["fused_score"] = round(fused[doc_id], 6)
        results.append(chunk)

    return results


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    SOURCES = [
        {
            "url": "https://www.coursicle.com/albany/?search=CINF+301&type=reviews",
            "source_type": "coursicle",
            "meta": {
                "source": "coursicle",
                "url": "https://www.coursicle.com/albany/?search=CINF+301&type=reviews",
                "topic_tag": "coursework",
                "date": "",
                "author_context": "current_student",
                "source_authority": 0.6,
            },
        },
        {
            "url": "https://talk.collegeconfidential.com/t/informatics-program/1792657",
            "source_type": "collegeconfidential",
            "meta": {
                "source": "collegeconfidential",
                "url": "https://talk.collegeconfidential.com/t/informatics-program/1792657",
                "topic_tag": "program_structure",
                "date": "",
                "author_context": "unknown",
                "source_authority": 0.5,
            },
        },
        
        {
        "url": "reddit_online.txt",
        "source_type": "manual",
        "meta": {
            "source": "reddit",
            "url": "https://www.reddit.com/r/ualbany/comments/o3u4yy/informatics_major/",
            "url": "https://www.reddit.com/r/ualbany/comments/aqo0i6/informatics_major_testimonies_has_anyone_gotten_a/",
            "topic_tag": "job_outcomes",
            "date": "2021-06",
            "author_context": "unknown",
            "source_authority": 0.5,
        },
    },
        
       {
        "url": "rmp.txt",
        "source_type": "manual",
        "meta": {
            "source": "ratemyprofessor",
            "url": "https://www.ratemyprofessors.com/professor/1918339",
            "topic_tag": "professor_quality",
            "date": "",
            "author_context": "current_student",
            "source_authority": 0.7,
        },
    },
        {
            "url": "https://www.niche.com/colleges/university-at-albany-suny/reviews/",
            "source_type": "niche",
            "meta": {
                "source": "niche",
                "url": "https://www.niche.com/colleges/university-at-albany-suny/reviews/",
                "topic_tag": "general",
                "date": "",
                "author_context": "unknown",
                "source_authority": 0.6,
            },
        },
            {
        "url": "coursicle_201.txt",
        "source_type": "manual",
        "meta": {
            "source": "coursicle",
            "url": "https://www.coursicle.com/albany/courses/CINF/201/",
            "topic_tag": "professor_quality",
            "date": "",
            "author_context": "current_student",
            "source_authority": 0.6,
        },
    },
        {
            "url": "https://www.reddit.com/r/ualbany/comments/adink4/informatics_bsonline/",
            "source_type": "reddit",
            "meta": {
                "source": "reddit",
                "url": "https://www.reddit.com/r/ualbany/comments/adink4/informatics_bsonline/",
                "topic_tag": "online_vs_inperson",
                "date": "2019-01",
                "author_context": "unknown",
                "source_authority": 0.5,
            },
        },
        {
            "url": "https://www.coursicle.com/albany/courses/CINF/200/",
            "source_type": "coursicle",
            "meta": {
                "source": "coursicle",
                "url": "https://www.coursicle.com/albany/courses/CINF/200/",
                "topic_tag": "coursework",
                "date": "",
                "author_context": "current_student",
                "source_authority": 0.6,
            },
        },
        
    ]

    all_chunks = []

    for entry in SOURCES:
        print(f"\n[loading] {entry['source_type']} — {entry['url'][:60]}")
        try:
            raw = load_source(entry["url"], entry["source_type"])
            if raw == "MANUAL_LOAD_REQUIRED":
                print(f"  → SKIPPED (manual load required)")
                continue
            chunks = chunk_text(raw, entry["meta"])
            print(f"  → {len(chunks)} chunks")
            all_chunks.extend(chunks)
            time.sleep(1)  # polite delay between requests
        except Exception as e:
            print(f"  → ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Total chunks across all sources: {len(all_chunks)}")
    print(f"{'='*60}\n")

    print("First 3 chunks:\n")
    for i, chunk in enumerate(all_chunks[:3]):
        print(f"--- Chunk {i+1} ---")
        print(f"text:             {chunk['text'][:200]}...")
        print(f"source:           {chunk['source']}")
        print(f"url:              {chunk['url']}")
        print(f"topic_tag:        {chunk['topic_tag']}")
        print(f"date:             {chunk['date']}")
        print(f"author_context:   {chunk['author_context']}")
        print(f"source_authority: {chunk['source_authority']}")
        print()

    # embed and store
    if all_chunks:
        print("Embedding and storing chunks...")
        embed_and_store(all_chunks)

        # test retrieval
        test_queries = [
            "did informatics graduates find jobs",
            "are the professors helpful and accessible",
            "is the online program worth it",
            "what do students say about CINF 301",
            "why do students avoid the informatics major",
        ]

        print("\nRetrieval test:\n")
        for query in test_queries:
            print(f"Query: {query}")
            results = retrieve(query, k=3)
            for r in results:
                print(f"  [{r['source']}] score={r['fused_score']}  {r['text'][:120]}...")
            print()