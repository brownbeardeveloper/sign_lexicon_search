from contextlib import asynccontextmanager
from typing import List
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import logging
import urllib.request

from core import _PROJECT_ROOT, _SIGNS_JSON
from core.config import load_config
from core.embedding import Embedding
from core.llm import LLM
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)

ml_models = {}


def _check_ollama(base_url: str, timeout: int = 5) -> None:
    """Ping the Ollama server. Raises SystemExit if unreachable."""
    try:
        urllib.request.urlopen(base_url, timeout=timeout)
        logger.info("✅ Ollama is reachable at %s", base_url)
    except Exception as e:
        raise SystemExit(
            f"\n❌ Ollama is not reachable at {base_url} ({e}).\n"
            f"   Start it with:  ollama serve\n"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()

    # --- Check that Ollama is running before doing anything else ---
    llm_config = config.get("LLM", {})
    _check_ollama(llm_config.get("base_url", "http://localhost:11434"))

    embedding = Embedding(config["Embedding"])
    embedding.load_model()
    store = VectorStore(embedding, config["Database"])
    index_dir = _PROJECT_ROOT / config.get("VectorStore", {}).get("index_dir", "data")
    
    index_filename = config.get("Database", {}).get("index_filename", "signs.index")
    if (index_dir / index_filename).exists():
        store.load(index_dir)
    else:
        store.build_from_json(_SIGNS_JSON)
        store.save(index_dir)
        
    ml_models["store"] = store
    ml_models["llm"] = LLM(llm_config)
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan, title="VectorDB Sign Language Search")
class SearchResult(BaseModel):
    id: str
    word: str
    variant_rank: int
    main_subject: str
    sub_subject: str
    link: str
    distance: float
    rank: int

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    response_time_ms: float

@app.get("/search", response_model=SearchResponse)
async def search_word(query: str, k: int = 5):
    """
    Search for words inside the loaded FAISS vector store database.
    """
    store: VectorStore = ml_models["store"]
    start_time = time.time()
    results = store.search(query, k=k)
    execution_time_ms = (time.time() - start_time) * 1000
    return SearchResponse(query=query, results=results, response_time_ms=execution_time_ms)

class SentenceSearchResponse(BaseModel):
    query: str
    results: dict[str, List[SearchResult]]
    response_time_ms: float

@app.get("/search/sentence", response_model=SentenceSearchResponse)
async def search_sentence(query: str, k: int = 5):
    """
    Search for a sentence using LLM lemmatization.
    Tries the full phrase first, then lemmatizes and searches individual base forms.
    """
    store: VectorStore = ml_models["store"]
    llm: LLM = ml_models["llm"]
    start_time = time.time()

    # Use LLM to get base forms, fall back to raw splitting
    base_forms = llm.extract_base_forms(query)
    logging.info(f"Base forms: {base_forms}")
    results = store.search_sentence(query, k=k, base_forms=base_forms)

    execution_time_ms = (time.time() - start_time) * 1000
    return SentenceSearchResponse(query=query, results=results, response_time_ms=execution_time_ms)

class GlossItem(BaseModel):
    gloss: str
    context: str | None = None
    spell: bool = False
    id: str | None = None
    link: str | None = None
    variant_rank: int | None = None

class GlossResponse(BaseModel):
    sentence: str
    glosses: List[GlossItem]
    response_time_ms: float

@app.get("/gloss", response_model=GlossResponse)
async def gloss_sentence(sentence: str):
    """
    Convert a Swedish sentence to TSP glosses using the LLM,
    then match each gloss against the sign database for ID and video link.
    """
    store: VectorStore = ml_models["store"]
    llm: LLM = ml_models["llm"]
    start_time = time.time()

    glosses: List[GlossItem] = []

    gloss_data = llm.glossify(sentence)

    if gloss_data is None:
        raise HTTPException(
            status_code=502,
            detail="LLM failed to produce valid TSP glosses. Check server logs for the raw LLM response.",
        )

    for g in gloss_data:
        word = g["word"]
        results = store.search(word, k=10)

        # Prefer results whose word shares a stem with the gloss
        query_lower = word.lower()
        stem_matches = [
            r for r in results
            if r["word"].lower().startswith(query_lower)
            or query_lower.startswith(r["word"].lower())
        ]
        pool = stem_matches if stem_matches else results
        best = min(pool, key=lambda x: x.get("variant_rank", 999)) if pool else {}

        glosses.append(GlossItem(
            gloss=word,
            context=g.get("context"),
            spell=g.get("spell", False),
            id=best.get("id"),
            link=best.get("link"),
            variant_rank=best.get("variant_rank"),
        ))

    execution_time_ms = (time.time() - start_time) * 1000
    return GlossResponse(sentence=sentence, glosses=glosses, response_time_ms=execution_time_ms)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # run with mamba run -n vector_db python main.py
