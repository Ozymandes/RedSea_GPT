from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_documents(pdf_dir: str):
    loader = PyPDFDirectoryLoader(pdf_dir)
    return loader.load()
