from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> HuggingFaceEmbeddings:
    """
    Get embeddings using Hugging Face model.

    Returns:
        HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings.
    """

    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs,
    )
    return embeddings
