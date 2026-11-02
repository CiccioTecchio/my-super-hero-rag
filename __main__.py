import argparse
from pathlib import Path

import faiss
import numpy as np
import tensorflow_hub as hub
import tensorflow_text

RESOURCES_DIR = Path(__file__).parent / "resources" / "rag_supereroi"
USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"


def read_documents(resources_dir: Path) -> list[str]:
    """Return the text of every Markdown document in the resource subfolders."""
    document_paths = sorted(
        path for path in resources_dir.glob("*/*.md") if path.is_file()
    )
    return [path.read_text(encoding="utf-8") for path in document_paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cerca la scheda di supereroi più pertinente a una domanda."
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="La domanda da cercare, per esempio: Chi è Bruce Wayne?",
    )
    return parser.parse_args()


def create_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a cosine-similarity index from document embeddings."""
    normalized_embeddings = np.ascontiguousarray(embeddings, dtype="float32")
    faiss.normalize_L2(normalized_embeddings)

    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)
    return index


def main():
    args = parse_args()
    question = " ".join(args.question)
    superheroes = read_documents(RESOURCES_DIR)

    if not superheroes:
        raise RuntimeError(f"Nessun documento trovato in {RESOURCES_DIR}")

    print("Caricamento di Universal Sentence Encoder...")
    embed = hub.load(USE_MODEL_URL)

    document_embeddings = embed(superheroes).numpy()
    index = create_index(document_embeddings)

    question_embedding = np.ascontiguousarray(
        embed([question]).numpy(), dtype="float32"
    )
    faiss.normalize_L2(question_embedding)

    scores, indices = index.search(question_embedding, k=1)
    best_match = int(indices[0][0])
    best_score = float(scores[0][0])

    print(f"\nDomanda: {question}")
    print(f"Similarità: {best_score:.4f}")
    print("\nDocumento più pertinente:\n")
    print(superheroes[best_match])


if __name__ == "__main__":
    main()
