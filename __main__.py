from pathlib import Path
from google import genai
from langchain_text_splitters import MarkdownHeaderTextSplitter
import argparse
import faiss
import logging
import numpy as np
import tensorflow_hub as hub
import tensorflow_text
import json
import re

logger = logging.getLogger(__name__)

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


def embed_documents(embed, documents: list[str]) -> np.ndarray:
    """Create embeddings for all documents."""
    return np.ascontiguousarray(embed(documents).numpy(), dtype="float32")


def embed_question(embed, question: str) -> np.ndarray:
    """Create the embedding for a question."""
    return np.ascontiguousarray(embed([question]).numpy(), dtype="float32")


def chunking(document: str) -> list[str]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")],
        strip_headers=False,
    )
    return [
        chunk.page_content
        for chunk in splitter.split_text(document)
        if "section" in chunk.metadata
    ]

def get_docs_chunks(documents:list[str]) -> list[str]:
    return [
        chunk
        for document in documents
        for chunk in chunking(document)
    ]

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

def answer_to_JSON(answer: str) -> dict[str, str | list[str]]:
    fenced_json = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        answer,
        re.DOTALL,
    )

    if fenced_json:
        answer = fenced_json.group(1)
    return json.loads(answer)

def ask_to_LLM(
    best_document: str,
    question: str,
) -> dict[str, str | list[str]]:
    prompt = f"""Answer the question based only on this context:
Context:
{best_document}
Question: {question}
Create a JSON with the following fields:
name: string //name of the character
superheroName: string //superhero name
universe: string //DC o Marvel
superpower: string[] //list of the superpower, if is a human list his/her characteristics
"""
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )
    return answer_to_JSON(response.text.strip())

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    question = " ".join(args.question)
    superheroes = read_documents(RESOURCES_DIR)
    if not superheroes:
        raise RuntimeError(f"Document not found in {RESOURCES_DIR}")
    logger.info("Loading Universal Sentence Encoder")
    embed = hub.load(USE_MODEL_URL)
    logger.info("Splitting documents into chunks")
    chunks = get_docs_chunks(superheroes)
    logger.info("Creating embeddings for %d chunks", len(chunks))
    document_embeddings = embed_documents(embed, chunks)
    logger.info("Creating the FAISS index")
    index = create_index(document_embeddings)
    logger.info("Embedding the question: %s", question)
    question_embedding = embed_question(embed, question)
    logger.info("Searching for the most relevant chunk")
    results = search_question(index, question_embedding, chunks)
    best_document, best_score = results[0]
    logger.info("Best match:\n%s", best_document)
    #logger.info("Question: %s", question)
    #logger.info("Similarity: %.4f", best_score)
    #logger.info("Most relevant document:\n%s", best_document)
    logger.info("Asking the LLM to create JSON from the retrieved information")
    print(ask_to_LLM(best_document, question))
    


if __name__ == "__main__":
    main()
