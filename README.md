# My Superhero RAG

First example of RAG usage based on [Building RAG Systems: From Zero to Hero](https://dev.to/gautamvhavle/building-production-rag-systems-from-zero-to-hero-2f1i).

The main concept is create a simple database of documents that contains info about the superheroes of DC and Marvel and search the information about them using a simple RAG.

> **Note:** In this first version, no vector database is used; embeddings are stored in memory.

## Resources and results

Information about the superheroes is available in the [resources](resources/) directory, while the results are available in [reports](reports/).

## Technologies used

|                                               Name                                               |        Version        |
| :----------------------------------------------------------------------------------------------: | :-------------------: |
|                                              Python                                              |         3.13          |
| [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3) |           2           |
|                                            Gemini AI                                             | gemini-3.5-flash-lite |

## Initial setup

Setup the venv with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# to use Gemini API you must subscribe to Google AI Studio and create your own API key and add it as env var with:
export GEMINI_API_KEY=<your_api_key>
```

## How to run

If you use task:

```bash
task main -- "Chi è Bruce Wayne?"
```

or:

```bash
.venv/bin/python __main__.py "Quali poteri possiede Thor?"
```

## Similarity matrix

Generate the cosine-similarity matrix between the questions in
`resources/rag_supereroi/domande_test.csv` and every document:

```bash
task similarity-matrix
```

The command writes two reports in `reports/`:

- `similarity_matrix.csv`, for further analysis;
- `similarity_matrix.html`, an interactive Plotly heatmap that highlights the
  best and expected documents for every question.
