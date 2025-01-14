import os
import argparse
from config import config
from typing import List
from pathlib import Path
from datetime import datetime
from data.document import Document
from models.embedding import OpenAIEmbeddingModel
from pipelines.rag import RAGPipeline


class RAGCLI:
    """Command-line interface for RAG pipeline operations."""

    def __init__(self):
        self.embedding_model = OpenAIEmbeddingModel()
        self.rag_pipeline = RAGPipeline(self.embedding_model)

    def __del__(self):
        """Clean up resources."""
        if self.rag_pipeline is not None:
            self.rag_pipeline.close()

    def add_documents(self,
                      file_paths: List[str],
                      chunk_size: int = config.OPENAI_CHUNK_SIZE,
                      chunk_overlap: int = 0):
        """Add and process documents to the vector store."""
        for file_path in file_paths:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Create document with metadata including filename and user-provided metadata
            document = Document(filepath=file_path,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                metadata={
                                    "filename": Path(file_path).name,
                                    "filepath": file_path,
                                    "chunk_size": chunk_size,
                                    "chunk_overlap": chunk_overlap,
                                    "added_at": datetime.now().isoformat()
                                })
            document.load()
            document.split()

            self.rag_pipeline.add_documents(document.chunks)

        print(f"Successfully processed {len(file_paths)} document(s)")

    def query(self, query: str, top_k: int = 5):
        """Query the vector store for relevant documents."""
        results = self.rag_pipeline.retrieve(query, k=top_k)

        print(f"\nTop {top_k} relevant documents:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n[{i}] Score: {score:.3f}")
            print("Content:")
            print(doc.page_content)
            print("\nMetadata:")
            for key, value in doc.metadata.items():
                print(f"{key}: {value}")
            print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description=
        "RAG Pipeline CLI - Manage and query documents using Retrieval Augmented Generation"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Add command
    add_parser = subparsers.add_parser(
        'add', help='Add documents to the vector store')
    add_parser.add_argument('files',
                            nargs='+',
                            help='Path(s) to document file(s) to add')
    add_parser.add_argument('--chunk-size',
                            type=int,
                            default=config.OPENAI_CHUNK_SIZE,
                            help='Size of text chunks (default: 300)')
    add_parser.add_argument('--chunk-overlap',
                            type=int,
                            default=0,
                            help='Overlap between chunks (default: 0)')

    # Query command
    query_parser = subparsers.add_parser('query',
                                         help='Query the vector store')
    query_parser.add_argument('query', help='Search query')
    query_parser.add_argument('--top-k',
                              type=int,
                              default=5,
                              help='Number of results to return (default: 5)')

    args = parser.parse_args()

    cli = RAGCLI()

    try:
        if args.command == 'add':
            cli.add_documents(args.files,
                              chunk_size=args.chunk_size,
                              chunk_overlap=args.chunk_overlap)
        elif args.command == 'query':
            cli.query(args.query, top_k=args.top_k)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
