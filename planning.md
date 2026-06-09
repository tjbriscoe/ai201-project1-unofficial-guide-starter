# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

--- Informatics Program Reviews at UAlbany 
## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 |Courside |A review of INF-301 Section @ Ualbany  |https://www.coursicle.com/albany/?search=CINF+301&type=reviews |
| 2 |CollegeConfidential |Student Review Forum of UAlbany's Informatics Program |https://talk.collegeconfidential.com/t/informatics-program/1792657 |
| 3 |Reddit |Rating the ROI of an Informatics degree at UAlbany |https://www.reddit.com/r/ualbany/comments/o3u4yy/informatics_major/ |
| 4 |Reddit |Questions and concerns about Informatics |https://www.reddit.com/r/ualbany/comments/lf06jl/questions_and_concerns_about_informatics_are/ |
| 5 |Reddit |Informatics Major Testimony |https://www.reddit.com/r/ualbany/comments/aqo0i6/informatics_major_testimonies_has_anyone_gotten_a/ |
| 6 |RateMyProfessor |Informatics Professor Reviews |https://www.ratemyprofessors.com/professor/1918339 |
| 7 |RateMyProfessor |Informatics Professor Reviews |https://www.ratemyprofessors.com/professor/3011538 |
| 8 |Niche| Student Reviews |https://www.niche.com/colleges/university-at-albany-suny/reviews/|
| 9 |Reddit |Information BS Online Program |https://www.reddit.com/r/ualbany/comments/adink4/informatics_bsonline/ |
| 10 |Coursicle |CINF 200 Course Assessment at UAlbany |https://www.coursicle.com/albany/courses/CINF/200/ |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**
150-250 tokens
**Overlap:**
25-40 tokens
**Reasoning:**

---Student reviews, Reddit comments, and RateMyProfessor entries are naturally short — most are 50–200 words. Using large chunks (512+) would bundle multiple unrelated opinions together, diluting signal. At 150–250 tokens you capture one complete thought or review per chunk without splitting a single person's opinion mid-sentence.
The overlap exists mainly for forum threads where a reply directly references the post above it — the overlap carries that connective tissue into the next chunk

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**
all-MiniLM-L6-v2
**Top-k:**
5-8 chunks

**Production tradeoff reflection:**
ll-MiniLM-L6-v2 is fast and lightweight but was trained on general web text. For student review language — informal phrasing, slang, acronyms like "CINF", "INF-301", "praxis" — a larger model picks up nuance it misses.
---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Scrapping inconsistency

2. Content gets deleted or changed 

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

---┌─────────────────────────────────────────────────────────────────┐
  │                    DOCUMENT INGESTION                           │
  │   Playwright · BeautifulSoup · Reddit API · pdfplumber          │
  └──────┬──────┬──────┬──────┬──────┬──────────────────────────────┘
         │      │      │      │      │
    Reddit(×4) Coursicle RMP  CC   Tableau PDF
    threads    reviews  (×2) forum  (survey)
         │      │      │      │      │
         └──────┴──────┴──────┴──────┘
                        │
                  clean + normalize
                  attach metadata
                        │
                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                       CHUNKING                                  │
  │   150–250 tokens · 25–40 token overlap                         │
  │                                                                 │
  │   reviews → 1 chunk each    │   PDF → section-level chunks     │
  │   split on \n\n boundary    │   question + response together   │
  └─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │               EMBEDDING + VECTOR STORE                          │
  │   all-MiniLM-L6-v2 (sentence-transformers)                      │
  │                                                                 │
  │   ┌──────────────────┐        ┌──────────────────┐             │
  │   │  dense index     │        │   BM25 index     │             │
  │   │  (Chroma/FAISS)  │        │   (rank-bm25)    │             │
  │   └──────────────────┘        └──────────────────┘             │
  └─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                      RETRIEVAL                                  │
  │                                                                 │
  │   query ──► dense search ──┐                                    │
  │         └─► BM25 search  ──┴──► RRF fusion ──► rerank ──► k=6  │
  │                                                                 │
  │   metadata filters: date · source_type · authority_score        │
  └─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                     GENERATION                                  │
  │   LangChain · OpenAI / Claude API                               │
  │                                                                 │
  │   map step: extract claim per chunk                             │
  │       │                                                         │
  │       ▼                                                         │
  │   reduce step: synthesize · surface conflicts · cite sources    │
  └─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
              answer + citations
              (Streamlit UI)

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**
Using the source table and chunking strategy below, implement two Python functions. First, load_source(url_or_path, source_type) that fetches and returns raw text from a Reddit thread, Coursicle page, RateMyProfessor page, or local PDF path. Use requests + BeautifulSoup for HTML sources and pdfplumber for the PDF. Second, chunk_text(raw_text, source_meta, chunk_size=200, overlap=30) that splits text on double newlines first, then enforces the token cap using the sentence-transformers tokenizer, and returns a list of dicts — each with text plus every field from source_meta.
**Milestone 4 — Embedding and retrieval:**
Given these 10 sample chunks [paste JSON], implement two functions. First, embed_and_store(chunks) that encodes each chunk's text field with all-MiniLM-L6-v2 via sentence-transformers and upserts into a ChromaDB collection with all metadata fields stored. Second, retrieve(query, k=6) that runs a dense cosine similarity search and a BM25 search using rank_bm25 on the same corpus, then combines scores with Reciprocal Rank Fusion and returns the top-k chunks sorted by fused score. No external reranker model — RRF is the final ranking step."

**Milestone 5 — Generation and interface:**
Given this sample retrieval output, implement generate_answer(query, chunks) using the Anthropic Python SDK. The function should run a map step first: for each chunk, send a prompt asking the model to extract the single most relevant claim to the query, or return 'not relevant' if none exists. Then run a reduce step: send all extracted claims in one prompt asking the model to synthesize a final answer, note any contradictions across sources, and flag if evidence is thin. Each claim in the final answer should include the source URL it came from. Return a dict with keys answer (string) and sources (list of URLs). Handle the case where fewer than 2 chunks return relevant claims by having the reduce step say evidence is insufficient rather than fabricating."