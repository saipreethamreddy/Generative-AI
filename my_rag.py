# """

# RAG (Retrieval-Augmented Generation) with Claude API

## WHAT IS RAG?

RAG = Give your LLM a "cheat sheet" from your own documents.
Instead of relying only on what the model was trained on,
we first SEARCH our PDF for relevant chunks, then pass those
chunks to Claude so it can answer accurately.

FLOW:
PDF → Split into Chunks → Embed (HuggingFace) → Store (FAISS)
↓
Question → Embed → Search FAISS → Top Chunks → Claude → Answer

INSTALL DEPENDENCIES (run this in your terminal first):
pip install anthropic langchain langchain-community \
langchain-huggingface faiss-cpu \
pypdf sentence-transformers
"""

import os
from anthropic import Anthropic

# ── LangChain components ──────────────────────────────────────────────────────

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =============================================================================

# STEP 1 – CONFIGURATION

# =============================================================================

# 🔑 Paste your Anthropic API key here (or set the env variable)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "your-anthropic-api-key-here")

# 📄 Path to your PDF file

PDF_PATH = "your_document.pdf"          # ← change this to your PDF path

# 🤖 Free HuggingFace embedding model (downloads automatically, no API key needed)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ✂️ Chunking settings

CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 50    # characters shared between neighbouring chunks
# (overlap helps avoid cutting off context at boundaries)

# 🔍 How many chunks to retrieve per question

TOP_K = 4

# =============================================================================

# STEP 2 – LOAD THE PDF

# =============================================================================

def load_pdf(pdf_path: str):
"""
PyPDFLoader reads the PDF page-by-page and returns a list of
LangChain Document objects, each containing:
- page_content : the raw text of that page
- metadata     : {"source": "file.pdf", "page": 0, ...}
"""
print(f"\n📄 Loading PDF: {pdf_path}")
loader = PyPDFLoader(pdf_path)
pages  = loader.load()
print(f"   ✅ Loaded {len(pages)} page(s)")
return pages

# =============================================================================

# STEP 3 – SPLIT INTO CHUNKS

# =============================================================================

def split_into_chunks(pages):
"""
RecursiveCharacterTextSplitter tries to split on paragraph breaks,
then sentences, then words – always preferring natural boundaries.

```
Why chunk?  Embedding models have token limits, and smaller, focused
chunks give better search precision than whole pages.
"""
print("\\n✂️  Splitting pages into chunks …")
splitter = RecursiveCharacterTextSplitter(
    chunk_size    = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP,
    separators    = ["\\n\\n", "\\n", ". ", " ", ""],   # try these in order
)
chunks = splitter.split_documents(pages)
print(f"   ✅ Created {len(chunks)} chunk(s)  "
      f"(size≈{CHUNK_SIZE} chars, overlap={CHUNK_OVERLAP})")
return chunks
```

# =============================================================================

# STEP 4 – EMBED & STORE IN FAISS

# =============================================================================

def build_vector_store(chunks):
"""
1. HuggingFaceEmbeddings converts each chunk into a vector
(a list of numbers that capture the chunk's meaning).
2. FAISS (Facebook AI Similarity Search) stores all vectors in
an index so we can find the nearest ones in milliseconds.

```
The model 'all-MiniLM-L6-v2' is small (80 MB), fast, and free.
It maps text → 384-dimensional vectors.
"""
print(f"\\n🔢 Loading embedding model: {EMBEDDING_MODEL}")
print("   (This downloads ~80 MB the first time – cached afterwards)")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("\\n📦 Building FAISS vector store …")
vector_store = FAISS.from_documents(chunks, embeddings)
print(f"   ✅ Stored {len(chunks)} vectors in FAISS")
return vector_store
```

# =============================================================================

# STEP 5 – BUILD THE RETRIEVER

# =============================================================================

def build_retriever(vector_store):
"""
We use MMR (Maximal Marginal Relevance) instead of plain similarity search.

```
MMR balances:
  - Relevance  : how similar a chunk is to the query
  - Diversity  : avoiding returning 4 near-identical chunks

fetch_k=20 → FAISS fetches 20 candidates
k=TOP_K    → MMR re-ranks and returns the best TOP_K
"""
print("\\n🔍 Setting up MMR retriever …")
retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k"       : TOP_K,
        "fetch_k" : 20,
    },
)
print(f"   ✅ Retriever ready  (returns top {TOP_K} diverse chunks per query)")
return retriever
```

# =============================================================================

# STEP 6 – ASK CLAUDE WITH RETRIEVED CONTEXT

# =============================================================================

def ask_claude(question: str, retriever, client: Anthropic) -> str:
"""
1. Embed the question and search FAISS for relevant chunks.
2. Build a prompt that includes those chunks as context.
3. Send the prompt to Claude and return its answer.
"""
# ── 6a. Retrieve relevant chunks ─────────────────────────────────────────
print(f"\n🔎 Searching for relevant chunks …")
relevant_docs = retriever.invoke(question)

```
# ── 6b. Format context block ──────────────────────────────────────────────
context_parts = []
for i, doc in enumerate(relevant_docs, 1):
    page = doc.metadata.get("page", "?")
    context_parts.append(
        f"[Chunk {i} | Page {page}]\\n{doc.page_content.strip()}"
    )
context = "\\n\\n---\\n\\n".join(context_parts)

print(f"   ✅ Retrieved {len(relevant_docs)} chunk(s) from the PDF")

# ── 6c. Build the prompt ──────────────────────────────────────────────────
prompt = f"""You are a helpful assistant. Answer the user's question using ONLY
```

the context excerpts provided below. If the answer is not in the context,
say "I couldn't find that information in the provided document."

CONTEXT FROM THE PDF:
{context}

USER QUESTION:
{question}

ANSWER:"""

```
# ── 6d. Call the Claude API ───────────────────────────────────────────────
print("🤖 Sending to Claude …")
response = client.messages.create(
    model      = "claude-sonnet-4-5",   # fast & capable
    max_tokens = 1024,
    messages   = [{"role": "user", "content": prompt}],
)

return response.content[0].text
```

# =============================================================================

# STEP 7 – MAIN: TIE EVERYTHING TOGETHER

# =============================================================================

def main():
print("=" * 60)
print("   RAG with Claude API + HuggingFace + FAISS")
print("=" * 60)

```
# ── Initialise Claude client ──────────────────────────────────────────────
client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Build the knowledge base ──────────────────────────────────────────────
pages        = load_pdf(PDF_PATH)
chunks       = split_into_chunks(pages)
vector_store = build_vector_store(chunks)
retriever    = build_retriever(vector_store)

# ── Optional: save & reload the FAISS index (avoids re-embedding) ─────────
# vector_store.save_local("faiss_index")
# vector_store = FAISS.load_local("faiss_index", embeddings,
#                                  allow_dangerous_deserialization=True)

# ── Interactive Q&A loop ──────────────────────────────────────────────────
print("\\n" + "=" * 60)
print("   ✅ Knowledge base is ready!  Start asking questions.")
print("   Type 'quit' or 'exit' to stop.")
print("=" * 60)

while True:
    question = input("\\n❓ Your question: ").strip()
    if not question:
        continue
    if question.lower() in {"quit", "exit"}:
        print("\\n👋 Goodbye!")
        break

    answer = ask_claude(question, retriever, client)
    print(f"\\n💬 Claude's answer:\\n{answer}")
    print("\\n" + "-" * 60)
```

if **name** == "**main**":
main()
