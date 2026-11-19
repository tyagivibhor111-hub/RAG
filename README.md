# RAG for CSV Data

A small project exploring how Retrieval-Augmented Generation (RAG) can be used to search structured data using natural language.

The project takes book records from a CSV file, converts them into text representations, and generates embeddings using NVIDIA's embedding model. These embeddings are stored in a FAISS index, which makes it possible to find books based on the meaning of a query rather than just matching exact words.

For example, instead of searching for a specific book title, you can enter something like `organic chemistry` and retrieve relevant books from the dataset.

## How it works

The workflow is divided into a few steps:

1. **Load the dataset** – Read book information from a CSV file using Pandas.
2. **Prepare the data** – Combine the author's name, book title, publisher, and publication year into a single text representation.
3. **Generate embeddings** – Use NVIDIA's `nv-embedqa-e5-v5` model to convert the text into numerical vectors.
4. **Build the search index** – Store the vectors in a FAISS index for similarity-based retrieval.
5. **Search books** – Convert the user's query into an embedding and retrieve the closest matches.
6. **Generate descriptions** – Use Llama 3.1 8B Instruct to generate short descriptions of the retrieved books.
7. **Display and summarize results** – Present the results in a readable format and create a short summary.
8. **Text-to-speech** – Use NVIDIA's TTS service to convert the generated summary into audio.

## Technologies used

- Python
- Pandas – Dataset handling
- NumPy – Numerical operations
- FAISS – Vector similarity search
- NVIDIA API – Embedding generation and text-to-speech
- Llama 3.1 8B Instruct – Book description generation
- OpenAI Python SDK – API client interface

## Project structure

```text
RAG/
│
├── RAG.ipynb          # Main notebook containing the implementation
├── data.csv           # Book dataset (provide your own)
├── metadata.npy       # Saved text metadata
├── faiss_index        # Saved FAISS vector index
└── audio.wav          # Generated speech output
```

The index, metadata, and audio files are generated when the corresponding notebook cells are executed.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/tyagivibhor111-hub/RAG.git
cd RAG
```

### 2. Install dependencies

```bash
pip install pandas numpy faiss-cpu openai
```

Run the notebook in Jupyter Notebook or Google Colab.

### 3. Configure your API key

The project uses NVIDIA API endpoints for embeddings and text generation.

Set your API key as an environment variable:

```bash
# PowerShell
$env:NVIDIA_API_KEY="your_api_key"
```

Configure the OpenAI-compatible client to use NVIDIA's API endpoint.

### 4. Add your dataset

Place your CSV file in the project directory.

The notebook expects the following columns:

- `Author's Name`
- `Title`
- `Publisher / Place`
- `Year`

Update the dataset filename in the notebook if necessary.

### 5. Run the notebook

Execute the notebook cells in order.

Once the FAISS index is created, you can search the dataset using natural-language queries.

## Example query

```python
query = "organic chemistry"
results = search_faiss(query)
```

The system retrieves relevant book records and generates descriptions for the results.

## What I learned

Working on this project helped me understand how different parts of a RAG pipeline fit together, particularly:

- Converting structured CSV records into searchable text.
- Generating and working with vector embeddings.
- Building a similarity search system with FAISS.
- Using an LLM after retrieval to generate useful descriptions.
- Connecting retrieval results with a text-to-speech pipeline.

## Future improvements

- Add support for larger datasets.
- Improve retrieval accuracy with better ranking strategies.
- Add a simple web interface for searching books.
- Cache embeddings to avoid generating them repeatedly.
- Handle missing or incomplete CSV records more reliably.

## Note

This project is an experimental implementation built around a notebook. It is intended for learning and exploring RAG concepts rather than serving as a production-ready search system.
