#!/usr/bin/env python3
"""Generate a question/document cosine-similarity matrix."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import tensorflow_hub as hub
import tensorflow_text  # noqa: F401 - registers TensorFlow Text custom ops


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RESOURCES_DIR = PROJECT_DIR / "resources" / "rag_supereroi"
DEFAULT_QUESTIONS_PATH = DEFAULT_RESOURCES_DIR / "domande_test.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "reports"
USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"


@dataclass(frozen=True)
class TestQuestion:
    text: str
    expected_documents: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera una matrice con la similarita coseno tra le domande di test "
            "e tutti i documenti del corpus."
        )
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="CSV delle domande di test (default: %(default)s)",
    )
    parser.add_argument(
        "--resources-dir",
        type=Path,
        default=DEFAULT_RESOURCES_DIR,
        help="Directory che contiene le cartelle dei documenti (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in cui scrivere CSV e HTML (default: %(default)s)",
    )
    parser.add_argument(
        "--model-url",
        default=USE_MODEL_URL,
        help="URL del modello TensorFlow Hub (default: %(default)s)",
    )
    return parser.parse_args()


def read_questions(path: Path) -> list[TestQuestion]:
    questions: list[TestQuestion] = []

    # utf-8-sig removes the BOM present at the beginning of domande_test.csv.
    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"domanda", "documenti_attesi"}
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Colonne mancanti in {path}: {missing}")

        for line_number, row in enumerate(reader, start=2):
            question = (row["domanda"] or "").strip()
            if not question:
                raise ValueError(f"Domanda vuota in {path}, riga {line_number}")

            expected_documents = tuple(
                document.strip()
                for document in (row["documenti_attesi"] or "").split(";")
                if document.strip()
            )
            questions.append(TestQuestion(question, expected_documents))

    if not questions:
        raise ValueError(f"Nessuna domanda trovata in {path}")

    return questions


def read_documents(resources_dir: Path) -> tuple[list[str], list[str]]:
    document_paths = sorted(
        path for path in resources_dir.glob("*/*.md") if path.is_file()
    )
    if not document_paths:
        raise ValueError(f"Nessun documento Markdown trovato in {resources_dir}")

    names = [path.relative_to(resources_dir).as_posix() for path in document_paths]
    contents = [path.read_text(encoding="utf-8") for path in document_paths]
    return names, contents


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Il modello ha prodotto almeno un embedding nullo")
    return vectors / norms


def calculate_scores(
    model: object,
    questions: list[TestQuestion],
    documents: list[str],
) -> np.ndarray:
    question_vectors = normalize(model([question.text for question in questions]).numpy())
    document_vectors = normalize(model(documents).numpy())
    return question_vectors @ document_vectors.T


def write_csv(
    output_path: Path,
    questions: list[TestQuestion],
    document_names: list[str],
    scores: np.ndarray,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "domanda",
                "documenti_attesi",
                "documento_migliore",
                "score_migliore",
                *document_names,
            ]
        )

        for question, row_scores in zip(questions, scores, strict=True):
            best_index = int(np.argmax(row_scores))
            writer.writerow(
                [
                    question.text,
                    ";".join(question.expected_documents),
                    document_names[best_index],
                    f"{row_scores[best_index]:.6f}",
                    *(f"{score:.6f}" for score in row_scores),
                ]
            )


def write_heatmap(
    output_path: Path,
    questions: list[TestQuestion],
    document_names: list[str],
    scores: np.ndarray,
) -> tuple[int, int]:
    evaluable = 0
    correct = 0
    custom_data = np.empty((*scores.shape, 3), dtype=object)
    shapes: list[dict[str, object]] = []

    for row_index, (question, row_scores) in enumerate(
        zip(questions, scores, strict=True)
    ):
        best_index = int(np.argmax(row_scores))
        best_document = document_names[best_index]
        expected = set(question.expected_documents)

        if expected:
            evaluable += 1
            if best_document in expected:
                correct += 1

        expected_label = ", ".join(question.expected_documents) or "nessun documento"
        for index, (document_name, score) in enumerate(
            zip(document_names, row_scores, strict=True)
        ):
            markers: list[str] = []
            if index == best_index:
                markers.append("migliore")
            if document_name in expected:
                markers.append("atteso")

            marker_label = f"<br>Stato: {', '.join(markers)}" if markers else ""
            custom_data[row_index, index] = (
                question.text,
                document_name,
                f"Atteso: {expected_label}{marker_label}",
            )

        for expected_index, document_name in enumerate(document_names):
            if document_name not in expected:
                continue
            shapes.append(
                {
                    "type": "rect",
                    "x0": expected_index - 0.47,
                    "x1": expected_index + 0.47,
                    "y0": row_index - 0.47,
                    "y1": row_index + 0.47,
                    "line": {"color": "#2563eb", "width": 3},
                    "fillcolor": "rgba(0, 0, 0, 0)",
                }
            )

        shapes.append(
            {
                "type": "rect",
                "x0": best_index - 0.38,
                "x1": best_index + 0.38,
                "y0": row_index - 0.38,
                "y1": row_index + 0.38,
                "line": {"color": "#111827", "width": 3},
                "fillcolor": "rgba(0, 0, 0, 0)",
            }
        )

    accuracy = (correct / evaluable * 100.0) if evaluable else 0.0
    figure = go.Figure(
        data=go.Heatmap(
            z=scores,
            x=list(range(len(document_names))),
            y=list(range(len(questions))),
            customdata=custom_data,
            colorscale="YlGn",
            zmin=0,
            zmax=1,
            colorbar={"title": "Similarità"},
            texttemplate="%{z:.3f}",
            hovertemplate=(
                "Domanda: %{customdata[0]}<br>"
                "Documento: %{customdata[1]}<br>"
                "Similarità: %{z:.6f}<br>"
                "%{customdata[2]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title={
            "text": (
                "Matrice di similarità domanda-documento"
                f"<br><sup>Top-1 negli attesi: {correct}/{evaluable} "
                f"({accuracy:.1f}%) · nero = migliore · blu = atteso</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        width=1500,
        height=max(800, len(questions) * 52),
        margin={"l": 420, "r": 100, "t": 210, "b": 80},
        shapes=shapes,
        xaxis={
            "title": "Documenti",
            "tickmode": "array",
            "tickvals": list(range(len(document_names))),
            "ticktext": document_names,
            "tickangle": -45,
            "side": "top",
        },
        yaxis={
            "title": "Domande",
            "tickmode": "array",
            "tickvals": list(range(len(questions))),
            "ticktext": [question.text for question in questions],
            "autorange": "reversed",
        },
    )
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    return correct, evaluable


def main() -> None:
    args = parse_args()
    questions_path = args.questions.resolve()
    resources_dir = args.resources_dir.resolve()
    output_dir = args.output_dir.resolve()

    questions = read_questions(questions_path)
    document_names, documents = read_documents(resources_dir)

    print(f"Caricamento del modello {args.model_url}...")
    model = hub.load(args.model_url)
    print(
        f"Calcolo degli score per {len(questions)} domande e "
        f"{len(documents)} documenti..."
    )
    scores = calculate_scores(model, questions, documents)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "similarity_matrix.csv"
    html_path = output_dir / "similarity_matrix.html"
    write_csv(csv_path, questions, document_names, scores)
    correct, evaluable = write_heatmap(
        html_path, questions, document_names, scores
    )

    accuracy = (correct / evaluable * 100.0) if evaluable else 0.0
    print(f"Top-1 negli attesi: {correct}/{evaluable} ({accuracy:.1f}%)")
    print(f"CSV:  {csv_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
