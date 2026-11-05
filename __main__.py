from pathlib import Path
import argparse
import logging
from src.chunking import get_docs_chunks
from src.embedding_use import (
    load_use,
    embed_documents,
    embed_question,
    create_index,
    search_question,
)

from src.ask_to_ai import (
    answer_to_JSON,
    ask_to_LLM
)

logger = logging.getLogger(__name__)

RESOURCES_DIR = Path(__file__).parent / "resources" / "rag_supereroi"

def read_documents(resources_dir: Path) -> list[str]:
    """Return the text of every Markdown document in the resource subfolders."""
    document_paths = sorted(
        path for path in resources_dir.glob("*/*.md") if path.is_file()
    )
    return [path.read_text(encoding="utf-8") for path in document_paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the superhero profile most relevant to a question."
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="The question to search for, for example: Who is Bruce Wayne?",
    )
    return parser.parse_args()

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
    embed = load_use()
    
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
