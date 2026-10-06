import json
import os
import re
import threading
from datetime import date, datetime
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import chromadb
import pandas as pd
from dateutil import parser as date_parser
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Any model id from console.groq.com -> Docs -> Models
GROQ_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b").strip().lower()

# Embeddings run inside this container (no external service needed)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

MODEL_CACHE = os.getenv("MODEL_CACHE", "/app/models")

# Groq's free tier allows only ~7,000 input tokens per minute, so keep
# prompts small: short chunks and few retrieved passages.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))

FINNHUB_API_KEY = os.getenv(
    "FINNHUB_API_KEY",
    "",
)

DOCUMENTS = Path("/app/documents")
CHROMA_PATH = "/app/chroma"
COLLECTION_NAME = "stock_documents"

# Same watchlist as the frontend, plus company names so the AI can
# recognise "how is Apple doing?" as a request about AAPL.
WATCHLIST = {
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "GOOGL": ["google", "alphabet"],
    "AMZN": ["amazon"],
    "TSLA": ["tesla"],
    "META": ["meta", "facebook"],
    "NVDA": ["nvidia"],
    "NFLX": ["netflix"],
    "AMD": ["advanced micro devices"],
    "INTC": ["intel"],
    "DIS": ["disney"],
    "PYPL": ["paypal"],
}

# Finnhub's free tier has no index data, so major indices are answered
# with the ETFs that track them.
INDEX_PROXIES = {
    "DIA": ["dow jones", "djia", "the dow"],
    "SPY": ["s&p 500", "s&p", "sp500", "sp 500"],
    "QQQ": ["nasdaq"],
}

ETF_NOTES = {
    "DIA": "SPDR Dow Jones ETF, tracks the DJIA",
    "SPY": "SPDR S&P 500 ETF, tracks the S&P 500",
    "QQQ": "Invesco QQQ ETF, tracks the Nasdaq-100",
}

QUOTE_TTL_SECONDS = 15


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def strip_api_prefix(request, call_next):
    """Lets a reverse proxy forward /api/* to this app without rewriting."""
    path = request.scope["path"]

    if path == "/api" or path.startswith("/api/"):
        request.scope["path"] = path[4:] or "/"

    return await call_next(request)


Settings.chunk_size = CHUNK_SIZE
Settings.chunk_overlap = 50

Settings.embed_model = FastEmbedEmbedding(
    model_name=EMBEDDING_MODEL,
    cache_dir=MODEL_CACHE,
)

Settings.llm = OpenAILike(
    model=GROQ_MODEL,
    api_base=GROQ_BASE_URL,
    api_key=GROQ_API_KEY or "missing",
    is_chat_model=True,
    context_window=32000,
    temperature=0.1,
    timeout=120,
)


# ---------------------------------------------------------------------------
# Quotes (Finnhub)
# ---------------------------------------------------------------------------

_quote_cache = {}
_quote_lock = threading.Lock()

EMPTY_QUOTE = {"price": None, "change": None, "percent": None}


def fetch_quote(symbol):
    now = time.time()

    with _quote_lock:
        cached = _quote_cache.get(symbol)

    if cached and now - cached[0] < QUOTE_TTL_SECONDS:
        return cached[1]

    params = urlencode({
        "symbol": symbol,
        "token": FINNHUB_API_KEY,
    })

    url = f"https://finnhub.io/api/v1/quote?{params}"

    try:
        with urlopen(url, timeout=10) as response:
            quote = json.loads(response.read().decode())

        # Finnhub returns c == 0 for unknown symbols
        if not quote.get("c"):
            return dict(EMPTY_QUOTE)

        result = {
            "price": quote.get("c"),
            "change": quote.get("d"),
            "percent": quote.get("dp"),
        }

        with _quote_lock:
            _quote_cache[symbol] = (now, result)

        return result

    except Exception as e:
        print(f"Quote fetch failed for {symbol}: {e}")

        # fall back to a stale value rather than blanking the UI
        if cached:
            return cached[1]

        return dict(EMPTY_QUOTE)


def get_quotes(symbols):
    symbols = list(dict.fromkeys(symbols))

    with ThreadPoolExecutor(max_workers=6) as pool:
        values = list(pool.map(fetch_quote, symbols))

    return dict(zip(symbols, values))


@app.get("/quotes")
def quotes(symbols: str):
    if not FINNHUB_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="FINNHUB_API_KEY is not configured",
        )

    wanted = [
        s.strip().upper()
        for s in symbols.split(",")
        if s.strip()
    ]

    return get_quotes(wanted)


# ---------------------------------------------------------------------------
# Documents / vector index
# ---------------------------------------------------------------------------

def has_documents():
    return DOCUMENTS.exists() and any(
        p.is_file() for p in DOCUMENTS.rglob("*")
    )


def get_collection(reset=False):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    return chroma_client.get_or_create_collection(COLLECTION_NAME)


MANIFEST = Path(CHROMA_PATH) / "manifest.json"


def documents_signature():
    return sorted(
        [str(p.relative_to(DOCUMENTS)), p.stat().st_size, int(p.stat().st_mtime)]
        for p in DOCUMENTS.rglob("*")
        if p.is_file()
    )


def create_index(rebuild=False):
    """Build (or load) the vector index.

    Re-embeds automatically when the files in /app/documents changed since
    the last build. Returns None when there are no documents.
    """
    DOCUMENTS.mkdir(parents=True, exist_ok=True)

    signature = documents_signature()

    try:
        saved = json.loads(MANIFEST.read_text())
    except Exception:
        saved = None

    manifest = {
        "embedding": EMBEDDING_MODEL,
        "chunk": CHUNK_SIZE,
        "files": signature,
    }

    if saved != manifest:
        rebuild = True

    collection = get_collection(reset=rebuild)

    vector_store = ChromaVectorStore(chroma_collection=collection)

    if not rebuild and collection.count() > 0:
        return VectorStoreIndex.from_vector_store(vector_store)

    if not signature:
        return None

    documents = SimpleDirectoryReader(
        str(DOCUMENTS),
        recursive=True,
    ).load_data()

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    built = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    MANIFEST.write_text(json.dumps(manifest))

    return built


index = None


@app.on_event("startup")
def startup():
    global index

    try:
        index = create_index()
    except Exception as e:
        print(f"Startup indexing skipped: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_provider": "groq",
        "model": GROQ_MODEL,
        "groq_configured": bool(GROQ_API_KEY),
        "embedding_model": EMBEDDING_MODEL,
        "finnhub_enabled": bool(FINNHUB_API_KEY),
        "documents_indexed": index is not None,
    }


@app.post("/index")
def rebuild_index():
    global index

    try:
        index = create_index(rebuild=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return {
        "status": "ok",
        "documents_indexed": index is not None,
    }


# ---------------------------------------------------------------------------
# AI chat (documents + live quotes)
# ---------------------------------------------------------------------------

# All-caps words that look like tickers but are not
TICKER_STOPWORDS = {
    "I", "A", "AI", "IT", "US", "USA", "ETF", "CEO", "CFO", "IPO", "PE",
    "EPS", "ROI", "GDP", "AM", "PM", "OK", "ALL", "AND", "THE", "FOR",
    "YTD", "DJIA", "API", "CSV", "PDF", "WHAT", "WHY", "HOW", "WHO",
    "NOT", "BUY", "SELL", "ARE", "IS", "TO", "OF", "IN", "ON", "AT",
    "BE", "OR", "IF", "ME", "MY", "DO", "SO", "NO", "UP", "AS", "BY",
    "AN", "HI", "TODAY", "NOW", "WAS", "CAN", "YOU", "HELP", "PRICE",
}


def find_symbols(question):
    """Find tickers: watchlist symbols/names (any case), $TICKER, or
    ALL-CAPS tickers such as PLTR."""
    found = []
    lowered = question.lower()

    for symbol, names in {**WATCHLIST, **INDEX_PROXIES}.items():
        # "DIS" is also a word, so only match it in capitals
        flags = 0 if symbol == "DIS" else re.IGNORECASE

        by_symbol = re.search(
            rf"(?<![A-Za-z])\$?{symbol}(?![A-Za-z])", question, flags
        )
        by_name = any(
            re.search(rf"\b{re.escape(n)}\b", lowered) for n in names
        )

        if by_symbol or by_name:
            found.append(symbol)

    extras = 0

    for m in re.finditer(
        r"(?<![A-Za-z])\$([A-Za-z]{1,5})(?![A-Za-z])"
        r"|(?<![A-Za-z$])([A-Z]{2,5})(?![A-Za-z])",
        question,
    ):
        dollar, caps = m.group(1), m.group(2)
        symbol = (dollar or caps).upper()

        if symbol in found or (caps and symbol in TICKER_STOPWORDS):
            continue

        if extras >= 3:
            break

        found.append(symbol)
        extras += 1

    return found


def build_market_context(question):
    if not FINNHUB_API_KEY:
        return "", []

    symbols = find_symbols(question)

    if not symbols:
        return "", []

    lines = []

    for symbol, q in get_quotes(symbols).items():
        if q["price"] is None:
            continue

        note = f" ({ETF_NOTES[symbol]})" if symbol in ETF_NOTES else ""

        lines.append(
            f"{symbol}{note}: price ${q['price']:.2f}, "
            f"change {q['change']:+.2f} ({q['percent']:+.2f}%)"
        )

    return "\n".join(lines), symbols


MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)

DATE_PATTERNS = [
    r"\b\d{4}-\d{1,2}-\d{1,2}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    rf"\b{MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?\b",
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH}\.?(?:,?\s*\d{{4}})?\b",
]


def find_dates(question):
    found = []
    default = datetime(date.today().year, 1, 1)

    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            text = re.sub(r"(\d)(st|nd|rd|th)", r"\1", match.group(0))

            try:
                found.append(date_parser.parse(text, default=default).date())
            except Exception:
                continue

    return list(dict.fromkeys(found))


def dataset_context(question):
    """Exact date lookup in CSV datasets (embeddings are poor at this)."""
    dates = find_dates(question)

    if not dates or not DOCUMENTS.exists():
        return ""

    blocks = []

    for path in sorted(DOCUMENTS.rglob("*.csv")):
        try:
            df = pd.read_csv(path)

            date_col = next(
                (
                    c for c in df.columns
                    if "date" in str(c).lower() or "time" in str(c).lower()
                ),
                df.columns[0],
            )

            df["_d"] = (
                pd.to_datetime(df[date_col], errors="coerce", utc=True)
                .dt.tz_localize(None)
                .dt.normalize()
            )
            df = df.dropna(subset=["_d"])

            if df.empty:
                continue

            blocks.append(
                f"Dataset {path.name} covers "
                f"{df['_d'].min():%Y-%m-%d} to {df['_d'].max():%Y-%m-%d}."
            )

            for d in dates:
                target = pd.Timestamp(d)
                rows = df[df["_d"] == target]
                label = f"Rows for {d:%Y-%m-%d}:"

                if rows.empty:
                    earlier = df[df["_d"] < target]

                    if earlier.empty:
                        blocks.append(f"No rows on or before {d:%Y-%m-%d}.")
                        continue

                    nearest = earlier["_d"].max()
                    rows = earlier[earlier["_d"] == nearest]
                    label = (
                        f"No row for {d:%Y-%m-%d} (weekend, holiday or past "
                        f"the end of the data). Nearest earlier row, "
                        f"{nearest:%Y-%m-%d}:"
                    )

                blocks.append(
                    label + "\n"
                    + rows.drop(columns="_d").head(5).to_csv(index=False)
                )

        except Exception as e:
            print(f"Dataset lookup failed for {path.name}: {e}")

    return "\n".join(blocks) + "\n\n" if blocks else ""


def system_prompt():
    return (
        "You are a stock trading assistant inside a fake-money trading app. "
        f"Today's date is {date.today():%A, %B %d, %Y}. "
        "Answer from the dataset rows and reference documents provided below "
        "first, and use the live market data for current prices. Quote "
        "numbers exactly as given and say where they came from. If the "
        "answer is not in the provided data, say so plainly instead of "
        "guessing. Never invent prices."
    )


@app.get("/chat")
def chat(q: str):
    global index

    question = q.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured",
        )

    market, symbols = build_market_context(question)
    dataset = dataset_context(question)

    extra = dataset

    if market:
        extra += f"Live market data (from Finnhub, just fetched):\n{market}\n\n"

    try:
        if index is None:
            index = create_index()

        if index is not None:
            # braces in the injected data must not be read as template vars
            safe_extra = extra.replace("{", "{{").replace("}", "}}")

            template = PromptTemplate(
                f"{system_prompt()}\n\n"
                f"{safe_extra}"
                "Reference documents:\n{context_str}\n\n"
                "Question: {query_str}\n"
                "Answer:"
            )

            query_engine = index.as_query_engine(
                similarity_top_k=TOP_K,
                response_mode="compact",
                text_qa_template=template,
            )

            answer = str(query_engine.query(question))
        else:
            prompt = (
                f"{system_prompt()}\n\n"
                f"{extra}"
                f"Question: {question}\n"
                "Answer:"
            )

            answer = Settings.llm.complete(prompt).text.strip()

    except Exception as e:
        print(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI request failed: {e}")

    # Qwen reasoning models may include their thinking in <think> tags
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    return {
        "answer": answer,
        "symbols": symbols,
        "used_documents": index is not None,
        "dataset_rows": bool(dataset),
    }
