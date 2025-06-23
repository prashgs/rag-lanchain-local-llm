from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


def split_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list:
    """
    Split documents into smaller chunks.

    Args:
        documents (list): A list of documents to be split.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between consecutive chunks.

    Returns:
        list: A list of split documents.
    """
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ":","."],

    )

    # Split the documents
    split_documents = text_splitter.split_documents(documents)
    return split_documents



