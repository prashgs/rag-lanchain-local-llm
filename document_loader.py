from langchain.document_loaders.pdf import PyPDFDirectoryLoader


def load_documents_from_directory(directory_path: str) -> list:
    """
    Load documents from a specified directory.

    Args:
        directory_path (str): The path to the directory containing PDF files.

    Returns:
        list: A list of loaded documents.
    """
    # Initialize the PDF directory loader
    pdf_loader = PyPDFDirectoryLoader(directory_path)

    # Load documents from the directory
    documents = pdf_loader.load()

    return documents
