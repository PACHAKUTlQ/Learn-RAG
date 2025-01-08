class DocumentChunker:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size

    def chunk_document(self, document):
        chunks = []
        for i in range(0, len(document), self.chunk_size):
            chunks.append(document[i:i + self.chunk_size])
        return chunks
