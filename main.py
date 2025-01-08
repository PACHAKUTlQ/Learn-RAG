import os
from chunking import DocumentChunker
from embedding import DocumentEmbedder
from retrieving import DocumentRetriever

def main():
    document = input("Enter the document: ")
    query = input("Enter the query: ")

    chunker = DocumentChunker()
    chunks = chunker.chunk_document(document)

    embedder = DocumentEmbedder()
    embeddings = embedder.embed_chunks(chunks)

    retriever = DocumentRetriever()
    results = retriever.retrieve_documents(query, embeddings)

    print("Top 5 relevant document chunks:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
