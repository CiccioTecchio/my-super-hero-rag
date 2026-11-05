import faiss
import numpy as np
import tensorflow_hub as hub
import tensorflow_text

def load_use():
    USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    return hub.load(USE_MODEL_URL)

def embed_documents(embed, documents: list[str]) -> np.ndarray:
    """Create embeddings for all documents."""
    return np.ascontiguousarray(embed(documents).numpy(), dtype="float32")


def embed_question(embed, question: str) -> np.ndarray:
    """Create the embedding for a question."""
    return np.ascontiguousarray(embed([question]).numpy(), dtype="float32")

def create_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a cosine-similarity index from document embeddings."""
    normalized_embeddings = np.ascontiguousarray(embeddings, dtype="float32")
    faiss.normalize_L2(normalized_embeddings)

    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)
    return index

def search_question(
    index: faiss.IndexFlatIP,
    question_embedding: np.ndarray,
    documents: list[str],
    k: int = 1,
) -> list[tuple[str, float]]:
    """Return the documents most similar to a question and their scores."""
    normalized_embedding = np.ascontiguousarray(
        question_embedding.copy(), dtype="float32"
    )
    faiss.normalize_L2(normalized_embedding)

    scores, indices = index.search(normalized_embedding, k=k)
    return [
        (documents[int(document_index)], float(score))
        for score, document_index in zip(scores[0], indices[0], strict=True)
        if document_index >= 0
    ]
