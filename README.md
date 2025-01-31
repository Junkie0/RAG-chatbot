# RAG Chatbot

## Overview

This is a Retrieval-Augmented Generation (RAG) chatbot that leverages FAISS for efficient document retrieval and a language model for generating responses. It is built using Flask for the backend and integrates a lightweight embedding model for document indexing and retrieval.

## Features

- Document-based retrieval using FAISS
- Chunking and embedding of textual data
- Query-based response generation using a fine-tuned TinyLlama model
- Flask-based API for easy interaction

## Installation

### Prerequisites

Ensure you have Python installed (recommended: Python 3.8 or later).

### Install Required Dependencies

Create an Virtual Env
```shell
  python -m venv RAGenv
```

Select this env to download and run all the dependencies

```sh
  pip install flask torch transformers sentence-transformers faiss-cpu nbconvert nbformat 'accelerate>=0.26.0'
```

## Project Structure

```
├── src/
│   ├── data.txt
│   ├── RAG.py
├── templates/
│   ├── index.html
├── app.py
├── README.md
```

## Usage

### Running the Chatbot

1. Ensure your text document is in `src/data.txt`.
2. Run the application:
   ```sh
   python app.py
   ```
3. The chatbot will start and load the document into FAISS.
4. Access the chatbot via the frontend or API endpoint `/ask`.

### API Usage

Send a POST request to `http://127.0.0.1:5000/ask` with JSON payload:

```json
{
  "question": "What is RAG?"
}
```

The response will be:

```json
{
  "answer": "Retrieval-Augmented Generation (RAG) combines document retrieval with generative models to provide more accurate responses.",
  "contexts": ["Context snippet 1", "Context snippet 2"]
}
```

## Implementation Details

- Uses **FAISS** for fast nearest-neighbor search.
- Embeddings generated using **SentenceTransformer (paraphrase-MiniLM-L3-v2)**.
- Language model used: **TinyLlama/TinyLlama-1.1B-Chat-v1.0**.
- Processes text into smaller chunks before embedding.
- The chatbot returns a response based on retrieved contexts.

## License

This project is open-source and available under the MIT License.

## Author

Anurag Potdar
