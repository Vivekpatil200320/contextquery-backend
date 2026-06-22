"""
ContextQuery RAG Evaluation Script
Measures: retrieval hit rate, reranker improvement, answer faithfulness
Compares: semantic-only vs hybrid (BM25 + semantic via RRF) retrieval modes

Run: python evals/eval_pipeline.py
"""

import asyncio
import json
import time
import httpx

# --- Config ---
BASE_URL = "http://localhost:8000"
RESULTS_FILE = "evals/results.json"
MODES = ["semantic", "hybrid"]

# --- Test cases ---
# Format: (question, expected_source_filename, keyword_that_must_appear_in_answer)
TEST_CASES = [
    (
        "What happened to the hummingbird Dikran found?",
        "Theme 7 Surviving the Elements (1).docx",
        "honey",
    ),
    (
        "What does Dikran do with the hummingbird at the end?",
        "Theme 7 Surviving the Elements (1).docx",
        "window",
    ),
    (
        "What is VDev Automations?",
        "linkedin-proflie-info.pdf",
        "lead",
    ),
    (
        "What is Vivek's role at VDev Automations?",
        "linkedin-proflie-info.pdf",
        "founder",
    ),
    (
        "What is Vivek's educational background?",
        "Vivek_Patil_Resume.docx",
        "MCA",
    ),
    (
        "What programming languages does Vivek know?",
        "Vivek_Patil_Resume.docx",
        "Python",
    ),
    (
        "What is the capital of France?",
        None,
        "not|don't|cannot|can't|outside|unable",
    ),
    (
        "Who wrote the Bible?",
        None,
        "not|don't|cannot|can't|outside|unable",
    ),
]


def check_retrieval_hit(sources: list[dict], expected_filename: str | None) -> bool:
    if expected_filename is None:
        return True  # out-of-context questions — not applicable
    filenames = [s.get("filename", "") for s in sources]
    return any(expected_filename in f for f in filenames)


def check_faithfulness(answer: str, keyword: str) -> bool:
    # Support pipe-separated alternatives — any match passes
    return any(k.strip().lower() in answer.lower() for k in keyword.split("|"))


async def run_query(client: httpx.AsyncClient, question: str, mode: str) -> dict:
    t0 = time.perf_counter()
    resp = await client.post(
        f"{BASE_URL}/api/query",
        json={"question": question, "retrieval_mode": mode},
        timeout=60.0,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000)
    data = resp.json()
    return {
        "answer": data.get("answer", ""),
        "sources": data.get("sources", []),
        "chunks_retrieved": data.get("chunks_retrieved", 0),
        "chunks_used": data.get("chunks_used", 0),
        "latency_ms": latency_ms,
    }


def _mode_summary(mode_results: list[dict]) -> dict:
    applicable = [r for r in mode_results if r["expected_source"] is not None]
    retrieval_hits = sum(1 for r in applicable if r["retrieval_hit"])
    faithfulness_hits = sum(1 for r in mode_results if r["faithfulness_pass"])
    avg_latency = sum(r["latency_ms"] for r in mode_results) / len(mode_results)
    return {
        "retrieval_precision": retrieval_hits / len(applicable) if applicable else 0,
        "retrieval_hits": retrieval_hits,
        "retrieval_applicable": len(applicable),
        "faithfulness_rate": faithfulness_hits / len(mode_results),
        "faithfulness_hits": faithfulness_hits,
        "total_cases": len(mode_results),
        "avg_latency_ms": round(avg_latency),
    }


async def main() -> None:
    print(f"\nContextQuery Eval — {len(TEST_CASES)} test cases × {len(MODES)} modes\n{'─' * 60}")

    all_results: dict[str, list[dict]] = {m: [] for m in MODES}

    async with httpx.AsyncClient() as client:
        print("Warming up backend...")
        try:
            await client.get(f"{BASE_URL}/health", timeout=10.0)
        except Exception:
            print("⚠ Backend not reachable — is uvicorn running?")
            return

        for i, (question, expected_source, keyword) in enumerate(TEST_CASES, 1):
            print(f"\n[{i}/{len(TEST_CASES)}] {question[:70]}")

            for mode in MODES:
                result = await run_query(client, question, mode)
                hit = check_retrieval_hit(result["sources"], expected_source)
                faithful = check_faithfulness(result["answer"], keyword)

                status = "✓" if (hit and faithful) else "✗"
                print(
                    f"  [{mode:8s}] {status}  retrieval={'hit ' if hit else 'MISS'}  "
                    f"faithful={'pass' if faithful else 'FAIL'}  "
                    f"{result['latency_ms']:5d}ms  "
                    f"chunks {result['chunks_retrieved']}→{result['chunks_used']}"
                )
                if not faithful:
                    print(f"           Answer: {result['answer'][:100]}...")

                all_results[mode].append({
                    "question": question,
                    "expected_source": expected_source,
                    "expected_keyword": keyword,
                    "retrieval_hit": hit,
                    "faithfulness_pass": faithful,
                    **result,
                })

    # Summary table
    print(f"\n{'─' * 60}")
    summaries = {m: _mode_summary(all_results[m]) for m in MODES}

    col = 20
    print(f"{'Metric':<25} {'semantic':>{col}} {'hybrid':>{col}}")
    print("─" * (25 + col * 2 + 2))

    s, h = summaries["semantic"], summaries["hybrid"]
    print(
        f"{'Retrieval precision':<25} "
        f"{s['retrieval_hits']}/{s['retrieval_applicable']} "
        f"({s['retrieval_precision']:.0%})".rjust(col) + " " +
        f"{h['retrieval_hits']}/{h['retrieval_applicable']} "
        f"({h['retrieval_precision']:.0%})".rjust(col)
    )
    print(
        f"{'Faithfulness rate':<25} "
        f"{s['faithfulness_hits']}/{s['total_cases']} "
        f"({s['faithfulness_rate']:.0%})".rjust(col) + " " +
        f"{h['faithfulness_hits']}/{h['total_cases']} "
        f"({h['faithfulness_rate']:.0%})".rjust(col)
    )
    print(
        f"{'Avg latency':<25} "
        f"{s['avg_latency_ms']}ms".rjust(col) + " " +
        f"{h['avg_latency_ms']}ms".rjust(col)
    )
    print(f"{'─' * 60}\n")

    output = {
        "summary": {m: summaries[m] for m in MODES},
        "results": {m: all_results[m] for m in MODES},
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
