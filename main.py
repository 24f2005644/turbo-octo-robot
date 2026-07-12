import os
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Semantic Search Top-K Ranking API")

# Reads your key from an environment variable — never hard-code it in the file.
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query_id: str
    query: str
    candidates: List[str]


class SearchResponse(BaseModel):
    ranking: List[int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Send a list of strings to the embedding model in ONE request and get back
    one vector per string, in the same order they were sent in.
    """
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


def cosine_similarities(query_vec: np.ndarray, candidate_vecs: np.ndarray) -> np.ndarray:
    """
    Cosine similarity = how similar the *direction* of two vectors is,
    ignoring their length. 1.0 = identical direction, 0 = unrelated,
    -1 = opposite.
    """
    query_unit = query_vec / np.linalg.norm(query_vec)
    candidate_norms = np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
    candidate_units = candidate_vecs / candidate_norms
    return candidate_units @ query_unit  # shape: (num_candidates,)


# ---------------------------------------------------------------------------
# The endpoint the grader calls
# ---------------------------------------------------------------------------
@app.post("/rank", response_model=SearchResponse)
def rank(request: SearchRequest):
    # Embed the query and all candidates together — this is one API call
    # instead of N+1, which is faster and cheaper.
    all_texts = [request.query] + request.candidates
    all_vectors = embed_texts(all_texts)

    query_vector = all_vectors[0]
    candidate_vectors = all_vectors[1:]

    scores = cosine_similarities(query_vector, candidate_vectors)

    # argsort gives ascending order (lowest first), so reverse it,
    # then keep only the top 3 indices.
    top_3 = np.argsort(scores)[::-1][:3]

    return SearchResponse(ranking=top_3.tolist())


# Simple root route so you (and the grader's uptime checks) can confirm
# the service is alive.
@app.get("/")
def health_check():
    return {"status": "ok"}
