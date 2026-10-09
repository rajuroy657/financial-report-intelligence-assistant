import os
import re
from pathlib import Path

import streamlit as st

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Optional dependencies for V3 and V4
try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = BASE_DIR / "faiss_financial_v2"
CHROMA_DIR = BASE_DIR / "chroma_financial_v3"
API_FILE = BASE_DIR / "api.txt"

CHROMA_COLLECTION = "financial_reports_v3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.5-flash-lite"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

REPORT_FILES = {
    "2023": DATA_DIR / "2023_Annual_Report.docx",
    "2024": DATA_DIR / "2024_Annual_Report.docx",
    "2025": DATA_DIR / "2025_AnnualReport.docx",
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Report Intelligence Assistant",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 750;
    }
    .subtitle {
        color: #777;
        font-size: 16px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">📊 Financial Report Intelligence Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Explore annual reports with RAG, semantic search, '
    'and optional reranking.</div>',
    unsafe_allow_html=True,
)


# ============================================================
# API KEY
# ============================================================

api_key = os.getenv("GOOGLE_API_KEY", "").strip()

if not api_key and API_FILE.exists():
    api_key = API_FILE.read_text(encoding="utf-8").strip()

if not api_key:
    st.error(
        "Gemini API key not found. Add it to api.txt or set "
        "the GOOGLE_API_KEY environment variable."
    )
    st.stop()


# ============================================================
# EMBEDDINGS AND LLM
# ============================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


@st.cache_resource
def load_llm(key):
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=key,
    )


with st.spinner("Loading embedding model and Gemini..."):
    embeddings = load_embeddings()
    llm = load_llm(api_key)


# ============================================================
# LOAD AND SPLIT ANNUAL REPORTS
# ============================================================

@st.cache_resource
def load_documents():
    all_docs = []

    for year, file_path in REPORT_FILES.items():
        if not file_path.exists():
            continue

        docs = Docx2txtLoader(str(file_path)).load()

        for doc in docs:
            doc.metadata["year"] = year
            doc.metadata["source"] = file_path.name
            doc.metadata["document_type"] = "annual_report"

        all_docs.extend(docs)

    return all_docs


@st.cache_resource
def create_chunks(_documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(_documents)


documents = load_documents()

if not documents:
    st.error(
        "No reports were loaded. Confirm that the DOCX files are in "
        "the data folder and their filenames match the configuration."
    )
    st.stop()

chunks = create_chunks(documents)


# ============================================================
# V2: FAISS
# ============================================================

@st.cache_resource
def load_or_create_faiss(_chunks, _embeddings):
    index_file = FAISS_DIR / "index.faiss"
    metadata_file = FAISS_DIR / "index.pkl"

    if index_file.exists() and metadata_file.exists():
        # Only load a FAISS index that you created or trust.
        return FAISS.load_local(
            str(FAISS_DIR),
            _embeddings,
            allow_dangerous_deserialization=True,
        )

    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    store = FAISS.from_documents(
        documents=_chunks,
        embedding=_embeddings,
    )
    store.save_local(str(FAISS_DIR))
    return store


def search_faiss(question, k=10, years=None):
    store = load_or_create_faiss(chunks, embeddings)

    # Retrieve a larger pool before applying year filters.
    candidates = store.similarity_search(
        question,
        k=min(max(k * 5, k), len(chunks)),
    )

    if years:
        candidates = [
            doc for doc in candidates
            if str(doc.metadata.get("year", "")) in years
        ]

    return candidates[:k]


# ============================================================
# V3 AND V4: LOAD EXISTING CHROMADB
# ============================================================

@st.cache_resource
def load_chroma_collection():
    if chromadb is None:
        raise RuntimeError(
            "ChromaDB is not importable in this Python environment. "
            "Install it using: python -m pip install chromadb"
        )

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {CHROMA_DIR}. "
            "Copy your existing V3 database into this location."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection(
            name=CHROMA_COLLECTION
        )
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{CHROMA_COLLECTION}' was not found in "
            f"{CHROMA_DIR}. Check your actual collection name."
        ) from exc

    if collection.count() == 0:
        raise RuntimeError(
            f"ChromaDB collection '{CHROMA_COLLECTION}' is empty."
        )

    return client, collection


def search_chroma_candidates(question, candidate_count=20, years=None):
    _, collection = load_chroma_collection()

    query_embedding = embeddings.embed_query(question)

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": min(candidate_count, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }

    # Supports one year or a multi-year comparison.
    if years:
        year_values = list(years)
        if len(year_values) == 1:
            query_args["where"] = {"year": year_values[0]}
        else:
            query_args["where"] = {"year": {"$in": year_values}}

    result = collection.query(**query_args)

    found_docs = []
    ids = result.get("ids", [[]])[0]
    texts = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for i, text in enumerate(texts):
        metadata = metadatas[i] if i < len(metadatas) else {}
        metadata = metadata or {}

        found_docs.append(
            Document(
                page_content=text or "",
                metadata={
                    **metadata,
                    "chroma_id": ids[i] if i < len(ids) else "",
                    "distance": (
                        distances[i] if i < len(distances) else None
                    ),
                },
            )
        )

    return found_docs


# ============================================================
# V3: CHROMADB SIMILARITY SEARCH
# ============================================================

def search_chroma(question, k=10, years=None):
    return search_chroma_candidates(
        question=question,
        candidate_count=k,
        years=years,
    )


# ============================================================
# V4: CHROMADB + CROSS-ENCODER RERANKING
# ============================================================

@st.cache_resource
def load_reranker():
    if CrossEncoder is None:
        raise RuntimeError(
            "V4 requires sentence-transformers. Install it using: "
            "python -m pip install sentence-transformers"
        )

    return CrossEncoder(RERANKER_MODEL)


def search_chroma_reranked(question, k=10, years=None):
    candidates = search_chroma_candidates(
        question=question,
        candidate_count=max(k * 4, 20),
        years=years,
    )

    if not candidates:
        return []

    reranker = load_reranker()

    pairs = [
        (question, doc.page_content)
        for doc in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    results = []

    for doc, score in ranked[:k]:
        doc.metadata["rerank_score"] = float(score)
        results.append(doc)

    return results


# ============================================================
# YEAR DETECTION
# ============================================================

def detect_years(question):
    # Handles single-year questions and questions comparing years.
    return set(re.findall(r"\b(2023|2024|2025)\b", question))


# ============================================================
# PROMPT AND OUTPUT PARSER
# ============================================================

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a financial report question-answering assistant.

Answer using ONLY the annual-report context below.

Rules:
- Do not use outside information or guess missing values.
- Preserve the numbers, currencies, and units as provided.
- Identify the report year when relevant.
- For a multi-year comparison, keep each year's values separate.
- If the answer is not supported by the context, say:
  "I could not find this information in the provided reports."

Context:
{context}

Question:
{question}

Answer:
""",
)

parser = StrOutputParser()
rag_chain = prompt | llm | parser


# ============================================================
# ROUTE QUESTION TO THE SELECTED VERSION
# ============================================================

def ask_report(question, version, k=10):
    years = detect_years(question)

    if version == "V2 - FAISS":
        results = search_faiss(question, k=k, years=years)

    elif version == "V3 - ChromaDB":
        results = search_chroma(question, k=k, years=years)

    elif version == "V4 - ChromaDB + Reranking":
        results = search_chroma_reranked(
            question,
            k=k,
            years=years,
        )

    else:
        raise ValueError(f"Unknown version: {version}")

    if not results:
        return (
            "I could not find relevant passages for the requested "
            "year or question in the available reports.",
            [],
        )

    context_parts = []

    for i, doc in enumerate(results, start=1):
        context_parts.append(
            f"[Source {i} | Year: {doc.metadata.get('year', 'Unknown')} "
            f"| File: {doc.metadata.get('source', 'Unknown')}]\n"
            f"{doc.page_content}"
        )

    answer = rag_chain.invoke(
        {
            "context": "\n\n".join(context_parts),
            "question": question,
        }
    )

    return answer.strip(), results


# ============================================================
# SIDEBAR: VERSION SELECTOR AND SETTINGS
# ============================================================

with st.sidebar:
    st.header("⚙️ RAG Version")

    version = st.selectbox(
        "Choose retrieval pipeline",
        [
            "V2 - FAISS",
            "V3 - ChromaDB",
            "V4 - ChromaDB + Reranking",
        ],
    )

    descriptions = {
        "V2 - FAISS": "FAISS similarity search",
        "V3 - ChromaDB": "Persistent ChromaDB similarity search",
        "V4 - ChromaDB + Reranking":
            "ChromaDB candidate retrieval + cross-encoder reranking",
    }

    st.caption(descriptions[version])

    st.divider()
    st.header("📚 Annual Reports")

    for year, file_path in REPORT_FILES.items():
        if file_path.exists():
            st.write(f"✅ {year}: {file_path.name}")
        else:
            st.write(f"⚠️ Missing: {file_path.name}")

    st.divider()
    st.write("**Models**")
    st.write(f"Embeddings: `{EMBEDDING_MODEL}`")
    st.write(f"LLM: `{LLM_MODEL}`")

    top_k = st.slider(
        "Final chunks sent to Gemini",
        min_value=3,
        max_value=20,
        value=10,
    )

    st.metric("Loaded documents", len(documents))
    st.metric("Text chunks", len(chunks))


# ============================================================
# MAIN QUESTION UI
# ============================================================

st.subheader("💬 Ask a financial question")

question = st.text_input(
    "Enter your question",
    placeholder="What was Microsoft's total revenue in fiscal year 2025?",
)

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "**Revenue**\n\n"
        "What was Microsoft's total revenue in fiscal year 2025?"
    )

with col2:
    st.info(
        "**Cloud**\n\n"
        "What was Microsoft Cloud revenue in fiscal year 2024?"
    )

with col3:
    st.info(
        "**Comparison**\n\n"
        "Compare Microsoft's revenue in 2024 and 2025."
    )


if st.button(
    "🔍 Ask Question",
    type="primary",
    use_container_width=True,
):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner(f"Running {version}..."):
                answer, results = ask_report(
                    question=question.strip(),
                    version=version,
                    k=top_k,
                )

            st.subheader("💡 Answer")
            st.markdown(answer)

            st.subheader("📚 Retrieved Sources")

            if not results:
                st.info("No passages were retrieved.")
            else:
                for i, doc in enumerate(results, start=1):
                    year = doc.metadata.get("year", "Unknown")
                    source = doc.metadata.get("source", "Unknown")

                    label = f"Source {i} | Year: {year} | {source}"

                    if "rerank_score" in doc.metadata:
                        label += (
                            f" | Rerank score: "
                            f"{doc.metadata['rerank_score']:.4f}"
                        )

                    with st.expander(label):
                        st.write(doc.page_content)

                        if doc.metadata.get("distance") is not None:
                            st.caption(
                                "Chroma distance: "
                                f"{doc.metadata['distance']:.4f}"
                            )

        except Exception as exc:
            st.error(f"{version} could not complete the request.")
            st.exception(exc)