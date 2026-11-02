# My Superhero RAG

First example of RAG usage based on [Building RAG Systems: From Zero to Hero](https://dev.to/gautamvhavle/building-production-rag-systems-from-zero-to-hero-2f1i)

## Initial setup

Setup the venv with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
