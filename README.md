# Learn-RAG
Learning of RAG, embedding and LLM

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/PACHAKUTlQ/Learn-RAG.git
    cd Learn-RAG
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up your environment variables. Create a `.env` file in the root directory of the project and add your OpenAI API key and base URL:
    ```sh
    OPENAI_API_KEY=your_openai_api_key_here
    BASE_URL=your_base_url_here
    ```

## Usage

1. Run the main script:
    ```sh
    python main.py
    ```

2. Follow the prompts to enter a document and a query. The script will chunk the document, embed the chunks, and retrieve the top 5 most relevant chunks based on the query.

## Files

- `chunking.py`: Contains the `DocumentChunker` class for chunking documents.
- `embedding.py`: Contains the `DocumentEmbedder` class for embedding document chunks.
- `retrieving.py`: Contains the `DocumentRetriever` class for performing vector search.
- `main.py`: Main script to handle user input and display results.

## Requirements

- `langchain`
- `openai`
