import json
import time
import os
from pathlib import Path
from datetime import datetime,timezone


from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from chain import build_chain

load_dotenv()

ROOT          = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = ROOT / "eval" / "eval_set.json"
RESULTS_PATH  = ROOT / "eval" / "eval_results.json"
SUMMARY_PATH  = ROOT / "eval" / "eval_summary.txt"

SLEEP_BETWEEN_QUERIES = 2   # seconds, avoids Groq rate limits
RAGAS_SAMPLE_SIZE     = 20  # run RAGAS on positive/partial_negative only

def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH) as f:
        return json.load(f)


def check_retrieval_recall(result: dict, item: dict) -> dict:
    #positive and partial negative questions should pass if any expected section appears in top chunks
    #negative pass if answer doesn't hallucinate
    #gate trigger pass if system refuses to answer

    question_type     = item["question_type"]
    expected_sections = item["expected_sections"]
    expected_drug     = item.get("expected_drug")

    if question_type == "gate_trigger":
        passed = result["refused"]
        return {
            "passed":  passed,
            "reason":  "correctly refused" if passed else "should have refused but did not",
        }

    if result["refused"]:
        return {
            "passed": False,
            "reason": "system refused but should have answered",
        }

    retrieved_sections = [
        src["section"].lower().replace(" ", "_")
        for src in result["sources"]
    ]
    retrieved_drugs = [
        src["drug_name"].lower()
        for src in result["sources"]
    ]

    if question_type == "negative":
        return {
            "passed": True,
            "reason": "answered appropriately for negative question",
        }

    if not expected_sections:
        return {
            "passed": True,
            "reason": "no expected sections defined",
        }

    # Check if any expected section appears in retrieved results
    section_hit = any(
        exp_sec in " ".join(retrieved_sections)
        for exp_sec in expected_sections
    )

    # Check if expected drug appears in retrieved results
    drug_hit = (
        expected_drug is None or
        any(
            expected_drug.lower() in drug
            for drug in retrieved_drugs
        )
    )

    passed = section_hit and drug_hit
    reason_parts = []
    if not section_hit:
        reason_parts.append(
            f"expected sections {expected_sections} not found in "
            f"retrieved {retrieved_sections}"
        )
    if not drug_hit:
        reason_parts.append(
            f"expected drug '{expected_drug}' not found in "
            f"retrieved {retrieved_drugs}"
        )
    if passed:
        reason_parts.append("correct drug and section retrieved")

    return {
        "passed": passed,
        "reason": "; ".join(reason_parts),
    }


def run_eval(chain, eval_set: list[dict]) -> list[dict]:
    #just runs questions through the chain and collects results

    results = []

    for i, item in enumerate(eval_set):
        print(f"[eval] {i+1}/{len(eval_set)}: {item['question'][:60]}...")

        try:
            result = chain.query(
                item["question"],
                run_grounding_check=False,  # grounding handled by RAGAS
            )
        except Exception as e:
            print(f"[eval] Error on question {i+1}: {e}")
            result = {
                "answer":   f"ERROR: {e}",
                "sources":  [],
                "refused":  False,
                "grounding": None,
                "metadata": {},
            }

        recall = check_retrieval_recall(result, item)

        results.append({
            "question":        item["question"],
            "question_type":   item["question_type"],
            "expected_answer": item["expected_answer"],
            "expected_drug":   item.get("expected_drug"),
            "expected_sections": item["expected_sections"],
            "generated_answer": result["answer"],
            "sources":          result["sources"],
            "refused":          result["refused"],
            "retrieval_recall": recall,
            "metadata":         result["metadata"],
        })

        time.sleep(SLEEP_BETWEEN_QUERIES)

    return results

def run_ragas(results: list[dict]) -> dict:
    scorable = [
        r for r in results
        if r["question_type"] in ("positive", "partial_negative")
        and not r["refused"]
        and r["generated_answer"]
        and not r["generated_answer"].startswith("ERROR")
    ][:RAGAS_SAMPLE_SIZE]

    if not scorable:
        print("[eval] No scorable results for RAGAS.")
        return {}

    print(f"[eval] Running RAGAS on {len(scorable)} questions...")

    # ragas defaults to openai API
    groq_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    ragas_llm   = LangchainLLMWrapper(groq_llm)
    ragas_embs  = LangchainEmbeddingsWrapper(hf_embeddings)

    for metric in [faithfulness, answer_relevancy, context_recall]:
        metric.llm        = ragas_llm
        metric.embeddings = ragas_embs

    ragas_data = {
        "question":     [r["question"] for r in scorable],
        "answer":       [r["generated_answer"] for r in scorable],
        "contexts":     [
            [src["full_text"] for src in r["sources"]]
            for r in scorable
        ],
        "ground_truth": [r["expected_answer"] for r in scorable],
    }

    dataset = Dataset.from_dict(ragas_data)
    scores  = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        raise_exceptions=False,
    )
    score_df = scores.to_pandas()
    return score_df.select_dtypes(include='number').mean().to_dict()


def compute_aggregate(results: list[dict]) -> dict:
    by_type = {}

    for r in results:
        qt = r["question_type"]
        if qt not in by_type:
            by_type[qt] = {"total": 0, "recall_passed": 0, "refused": 0}
        by_type[qt]["total"] += 1
        if r["retrieval_recall"]["passed"]:
            by_type[qt]["recall_passed"] += 1
        if r["refused"]:
            by_type[qt]["refused"] += 1

    summary = {}
    for qt, counts in by_type.items():
        summary[qt] = {
            "total":          counts["total"],
            "recall_pass_rate": round(
                counts["recall_passed"] / counts["total"], 3
            ),
            "refusal_rate":   round(
                counts["refused"] / counts["total"], 3
            ),
        }

    return summary


def write_summary(
    results: list[dict],
    aggregate: dict,
    ragas_scores: dict,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()

    lines = [
        "FDA RAG — Evaluation Summary",
        f"Run at: {timestamp}",
        f"Total questions: {len(results)}",
        "",
        "── Retrieval Recall by Question Type ──",
    ]

    for qt, stats in aggregate.items():
        lines.append(
            f"  {qt:<20} "
            f"recall: {stats['recall_pass_rate']:.1%}  "
            f"refusals: {stats['refusal_rate']:.1%}  "
            f"(n={stats['total']})"
        )

    lines += ["", "── RAGAS Scores (positive + partial_negative) ──"]

    if ragas_scores:
        for metric, score in ragas_scores.items():
            if isinstance(score, float):
                lines.append(f"  {metric:<25} {score:.4f}")
    else:
        lines.append("  No RAGAS scores computed.")

    lines += ["", "── Per-Question Retrieval Recall ──"]

    for r in results:
        status = "PASS" if r["retrieval_recall"]["passed"] else "FAIL"
        refused = " [REFUSED]" if r["refused"] else ""
        lines.append(
            f"  [{status}]{refused} "
            f"({r['question_type']}) "
            f"{r['question'][:55]}..."
        )
        if not r["retrieval_recall"]["passed"]:
            lines.append(f"         reason: {r['retrieval_recall']['reason']}")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(SUMMARY_PATH, "w") as f:
        f.write(summary_text)

    print(f"\n[eval] Summary written to {SUMMARY_PATH}")

def main():
    eval_set = load_eval_set()
    print(f"[eval] Loaded {len(eval_set)} questions from {EVAL_SET_PATH}")

    # use cached results if we have it
    if RESULTS_PATH.exists():
        print(f"[eval] Found existing results at {RESULTS_PATH} — skipping queries.")
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        chain   = build_chain()
        results = run_eval(chain, eval_set)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] Full results written to {RESULTS_PATH}")

    aggregate    = compute_aggregate(results)
    ragas_scores = run_ragas(results)
    write_summary(results, aggregate, ragas_scores)

if __name__ == "__main__":
    main()