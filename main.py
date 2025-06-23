from build_vector_db import add_to_vector_db
from document_loader import load_documents_from_directory
from get_embeddings import get_embeddings
from text_splitter import split_documents
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from chromadb.config import Settings
import chromadb


DATA_FOLDER = "data"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

Question: {question}
"""

query_text = """
What is Amazon SES?
"""

CHROMA_PATH = "db"
COLLECTION_NAME = "all-mpnet-base-v2"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def main():
    documents = load_documents_from_directory(DATA_FOLDER)
    print(f"Loaded {len(documents)} documents from {DATA_FOLDER}")

    chunks = split_documents(documents)
    print(f"Splitting {len(documents)} documents into chunks {len(chunks)}...")
    add_to_vector_db(chunks, collection_name=COLLECTION_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))

    embedding_function = get_embeddings()
    db = Chroma(
        client=chroma_client,
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME,
    )

    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="gemma3:1b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


def main_hf():
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings

    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    texts = ["Hello, world!", "How are you?"]
    embeddings = hf_embeddings.embed_documents(texts)
    print(embeddings)


if __name__ == "__main__":
    main()
