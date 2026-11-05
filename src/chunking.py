from langchain_text_splitters import MarkdownHeaderTextSplitter


def chunking(document: str) -> list[str]:
    """Split a Markdown document into sections, preserving their headers."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")],
        strip_headers=False,
    )
    return [
        chunk.page_content
        for chunk in splitter.split_text(document)
        if "section" in chunk.metadata
    ]


def get_docs_chunks(documents: list[str]) -> list[str]:
    """Return the chunks extracted from all documents."""
    return [chunk for document in documents for chunk in chunking(document)]
