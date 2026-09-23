
import json
import re
import time
from pathlib import Path
from typing import Dict, List


def extract_numbers(text: str) -> List[float]:
    if not text:
        return []

    normalized = text.replace(",", "")

    matches = re.findall(
        r"(?<!\w)(?:\$)?\d+(?:\.\d+)?",
        normalized
    )

    values = []

    for value in matches:
        try:
            values.append(float(value.replace("$", "")))
        except ValueError:
            pass

    return values


def numerical_accuracy(
    answer: str,
    expected_value: float,
    tolerance: float = 0.001
) -> float:

    numbers = extract_numbers(answer)

    if not numbers:
        return 0.0

    for number in numbers:

        if expected_value == 0:
            if abs(number) <= tolerance:
                return 1.0

        else:
            relative_error = (
                abs(number - expected_value)
                / abs(expected_value)
            )

            if relative_error <= tolerance:
                return 1.0

    return 0.0


def page_match(
    retrieved_chunks: List[Dict],
    expected_page: int
) -> float:

    pages = [
        chunk.get("page")
        for chunk in retrieved_chunks
    ]

    return 1.0 if expected_page in pages else 0.0


def reciprocal_rank(
    retrieved_chunks: List[Dict],
    expected_page: int
) -> float:

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        if chunk.get("page") == expected_page:
            return 1.0 / rank

    return 0.0


def save_results(results, output_path):

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )


def run_single_rag(
    rag,
    question,
    expected_value,
    expected_page,
    retries=2
):

    last_error = None

    for attempt in range(1, retries + 1):

        try:

            start = time.perf_counter()

            output = rag.answer(
                question,
                top_k=5
            )

            latency = (
                time.perf_counter()
                - start
            )

            return {
                "success": True,
                "answer": output["answer"],
                "model": output["model"],
                "latency_seconds": round(
                    latency,
                    4
                ),
                "numerical_accuracy":
                    numerical_accuracy(
                        output["answer"],
                        expected_value
                    ),
                "source_page_retrieved":
                    page_match(
                        output["retrieved_chunks"],
                        expected_page
                    ),
                "mrr":
                    reciprocal_rank(
                        output["retrieved_chunks"],
                        expected_page
                    ),
                "retrieved_chunks": [
                    {
                        "page": c.get("page"),
                        "score": c.get("score"),
                        "chunk_type":
                            c.get("chunk_type"),
                        "text": c.get("text")
                    }
                    for c in output[
                        "retrieved_chunks"
                    ]
                ]
            }

        except Exception as e:

            last_error = str(e)

            print(
                f"    Attempt {attempt}/{retries} failed:"
            )
            print(
                f"    {last_error}"
            )

            if attempt < retries:
                print(
                    "    Waiting before retry..."
                )
                time.sleep(8)

    return {
        "success": False,
        "error": last_error
    }


def run_benchmark(
    benchmark_questions,
    naive_rag,
    table_rag,
    output_path="results/benchmark_results.json"
):

    results = []

    output = Path(output_path)

    # ---------------------------------------------------------
    # LOAD PREVIOUS RESULTS
    # ---------------------------------------------------------

    if output.exists():

        try:

            with open(
                output,
                "r",
                encoding="utf-8"
            ) as f:

                results = json.load(f)

            print(
                f"Loaded {len(results)} previously "
                f"completed questions."
            )

        except Exception:

            print(
                "Existing results file could not be read."
            )

            results = []

    completed_ids = {
        r["id"]
        for r in results
        if r.get("status") == "completed"
    }

    total_questions = len(
        benchmark_questions
    )

    print("=" * 80)
    print("FINAUDIT RAGBENCH")
    print("RESUMABLE BENCHMARK EXECUTION")
    print("=" * 80)

    print(
        f"Questions: {total_questions}"
    )

    print(
        f"Already completed: "
        f"{len(completed_ids)}"
    )

    # ---------------------------------------------------------
    # PROCESS QUESTIONS
    # ---------------------------------------------------------

    for index, question_data in enumerate(
        benchmark_questions,
        start=1
    ):

        question_id = question_data["id"]

        if question_id in completed_ids:

            print(
                f"\n[{index}/{total_questions}] "
                f"{question_id} already completed. Skipping."
            )

            continue

        question = question_data["question"]

        print(
            f"\n[{index}/{total_questions}] "
            f"{question}"
        )

        expected_value = question_data[
            "expected_value"
        ]

        expected_page = question_data[
            "source_page"
        ]

        # -----------------------------------------------------
        # NAIVE
        # -----------------------------------------------------

        print(
            "  Running Naive RAG..."
        )

        naive = run_single_rag(
            naive_rag,
            question,
            expected_value,
            expected_page
        )

        if not naive["success"]:

            print(
                "\nNaive RAG failed after retries."
            )

            print(
                "Saving previous results."
            )

            save_results(
                results,
                output_path
            )

            raise RuntimeError(
                "Naive RAG temporarily unavailable. "
                "Previous benchmark results were saved."
            )

        print(
            "  Naive numerical accuracy:",
            naive["numerical_accuracy"]
        )

        # -----------------------------------------------------
        # TABLE-AWARE
        # -----------------------------------------------------

        print(
            "  Running Table-Aware RAG..."
        )

        table = run_single_rag(
            table_rag,
            question,
            expected_value,
            expected_page
        )

        if not table["success"]:

            print(
                "\nTable-Aware RAG failed after retries."
            )

            print(
                "Saving previous results."
            )

            save_results(
                results,
                output_path
            )

            raise RuntimeError(
                "Table-Aware RAG temporarily unavailable. "
                "Previous benchmark results were saved."
            )

        print(
            "  Table-aware numerical accuracy:",
            table["numerical_accuracy"]
        )

        # -----------------------------------------------------
        # COMPLETE RECORD
        # -----------------------------------------------------

        result = {
            "id": question_data["id"],
            "status": "completed",
            "question": question,
            "category": question_data["category"],
            "reference_answer":
                question_data["reference_answer"],
            "expected_value": expected_value,
            "unit": question_data["unit"],
            "year": question_data["year"],
            "source_page": expected_page,
            "naive": naive,
            "table_aware": table
        }

        results.append(result)

        # -----------------------------------------------------
        # SAVE IMMEDIATELY
        # -----------------------------------------------------

        save_results(
            results,
            output_path
        )

        print(
            "  ✓ Question saved."
        )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    print(
        "Completed:",
        len(results),
        "/",
        total_questions
    )

    print(
        "Results:",
        output_path
    )

    return results
