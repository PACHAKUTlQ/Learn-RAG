from config import config
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


class Document:

    def __init__(self,
                 filepath: str,
                 chunk_size: int = config.OPENAI_CHUNK_SIZE,
                 chunk_overlap: int = 0,
                 metadata: dict = None):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.raw_documents = None
        self.chunks = None
        self.metadata = metadata or {}

    def load(self):
        loader = TextLoader(self.filepath)
        self.raw_documents = loader.load()
        # Add metadata to each document
        for doc in self.raw_documents:
            doc.metadata.update(self.metadata)
            doc.metadata["source"] = self.filepath

    def split(self):
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size,
                                              chunk_overlap=self.chunk_overlap)
        self.chunks = text_splitter.split_documents(self.raw_documents)
