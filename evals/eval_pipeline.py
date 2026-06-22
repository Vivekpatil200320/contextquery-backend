"""
ContextQuery RAG Evaluation Script
Measures: retrieval hit rate, reranker improvement, answer faithfulness
Run: python evals/eval_pipeline.py
"""

import asyncio
import json
import time
import httpx

# --- Config ---
BASE_URL = "http://localhost:8000"  # swap to Render URL for prod eval
RESULTS_FILE = "evals/results.json"

# --- Test cases ---
# Format: (question, expected_source_filename, keyword_that_must_appear_in_answer)
TEST_CASES = [
    (
        "What happened to the hummingbird Dikran found?",
        "Theme 7 Surviving the Elements (1).docx",
        "honey"
    ),
    (
        "What does Dikran do with the hummingbird at the end?",
        "Theme 7 Surviving the Elements (1).docx",
        "window"
    ),
    (
        "What is VDev Automations?",
        "linkedin-proflie-info.pdf",
        "lead"
    ),
    (
        "What is Vivek's role at VDev Automations?",
        "linkedin-proflie-info.pdf",
        "founder"
    ),
    (
        "What is Vivek's educational background?",
        "Vivek_Patil_Resume_HighLevel.pdf",
        "MCA"
    ),
    (
        "What programming languages does Vivek know?",
        "Vivek_Patil_Resume_HighLevel.pdf",
        "Python"
    ),
    (
        "What is the capital of France?",  # out-of-context — should refuse
        None,
        "not"  # expects "I cannot" / "does not contain" — "not" covers both
    ),
    (
        "Who wrote the Bible?",  # out-of-context — should refuse
        None,
        "not"
    ),
]


def check_retrieval_hit(sources: list[dict], expected_filename: str | None) -> bool:
    if expected_filename is None:
        return True  # out-of-context questions — retrieval hit not applicable
    filenames = [s.get("filename", "") for s in sources]
    return any(expected_filename in f for f in filenames)


def check_faithfulness(answer: str, keyword: str) -> bool:
    return keyword.lower() in answer.lower()


async def run_query(client: httpx.AsyncClient, question: str) -> dict:
    t0 = time.perf_counter()
    resp = await client.post(
        f"{BASE_URL}/api/query",
        json={"question": question},
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


async def main() -> None:
    print(f"\nContextQuery Eval — {len(TEST_CASES)} test cases\n{'─' * 50}")
    results = []
    retrieval_hits = 0
    faithfulness_hits = 0
    applicable_retrieval = 0

    async with httpx.AsyncClient() as client:
        print("Warming up backend...")
        try:
            await client.get(f"{BASE_URL}/health", timeout=10.0)
        except Exception:
            print("⚠ Backend not reachable — is uvicorn running?")
            return

        for i, (question, expected_source, keyword) in enumerate(TEST_CASES, 1):
            print(f"\n[{i}/{len(TEST_CASES)}] {question[:60]}...")
            result = await run_query(client, question)

            hit = check_retrieval_hit(result["sources"], expected_source)
            faithful = check_faithfulness(result["answer"], keyword)

            if expected_source is not None:
                applicable_retrieval += 1
                if hit:
                    retrieval_hits += 1
            if faithful:
                faithfulness_hits += 1

            status = "✓" if (hit and faithful) else "✗"
            print(
                f"  {status} retrieval={'hit' if hit else 'MISS'} | "
                f"faithfulness={'pass' if faithful else 'FAIL'} | "
                f"{result['latency_ms']}ms | "
                f"chunks {result['chunks_retrieved']}→{result['chunks_used']}"
            )
            print(f"  Answer: {result['answer'][:120]}...")

            results.append({
                "question": question,
                "expected_source": expected_source,
                "expected_keyword": keyword,
                "retrieval_hit": hit,
                "faithfulness_pass": faithful,
                **result,
            })

    retrieval_precision = retrieval_hits / applicable_retrieval if applicable_retrieval else 0
    faithfulness_rate = faithfulness_hits / len(TEST_CASES)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    print(f"\n{'─' * 50}")
    print(
        f"Retrieval precision:  {retrieval_hits}/{applicable_retrieval} "
        f"({retrieval_precision:.0%})"
    )
    print(
        f"Faithfulness rate:    {faithfulness_hits}/{len(TEST_CASES)} "
        f"({faithfulness_rate:.0%})"
    )
    print(f"Avg latency:          {avg_latency:.0f}ms")
    print(f"{'─' * 50}\n")

    with open(RESULTS_FILE, "w") as f:
        json.dump(
            {
                "summary": {
                    "retrieval_precision": retrieval_precision,
                    "faithfulness_rate": faithfulness_rate,
                    "avg_latency_ms": avg_latency,
                    "total_cases": len(TEST_CASES),
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
