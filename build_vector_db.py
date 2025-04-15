from get_embeddings import get_embeddings
from langchain.vectorstores.chroma import Chroma

CHROMA_PATH = "db"
DATA_PATH = "data"


def add_to_vector_db(chunks: list, collection_name: str = "default"):
    """
    Add documents to the vector database.

    Args:
        documents (list): A list of documents to add.
        collection_name (str): The name of the collection in the vector database.
    """
    # Initialize the vector store
    embeddings = get_embeddings()
    vector_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    existing_items = vector_db.get(include=[])
    existing_ids = set(existing_items["ids"])
    chunks_with_ids = add_chunk_ids(chunks)

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vector_db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def get_chunks_from_documents(documents: list) -> list:
    """
    Split documents into chunks.

    Args:
        documents (list): A list of documents to split.

    Returns:
        list: A list of document chunks.
    """
    # For simplicity, let's assume each document is a string and we split it into chunks of 100 characters
    chunks = [doc[i : i + 100] for doc in documents for i in range(0, len(doc), 100)]
    return chunks


def add_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks
