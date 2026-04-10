"""RAG context quality benchmark.

Measures the token cost of RAG-retrieved chunks (in Cursor/Claude tokens)
and outputs a rubric template for manual A/B comparison in Cursor.
Now also reports session escalation and document grouping effects.
"""

import os
import sys

import tiktoken

from agents_rag.searcher import Searcher

QUESTION = "What are the known blockers and risks for migrating system-probe to Bazel?"

RUBRIC = [
    ("ebpf_prog Starlark rule", ["ebpf_prog", "starlark"]),
    ("GCC cannot compile eBPF, only Clang/LLVM", ["gcc", "clang"]),
    ("Dependency cycles block 73 of 190 Go modules", ["dependency cycle", "73", "cycle"]),
    ("Hybrid build is not feasible", ["hybrid", "not feasible"]),
    ("Missing BUILD files in gazelle-excluded dirs", ["missing build", "gazelle"]),
    ("Direct deps in un-excluded trees", ["un-excluded", "direct dep"]),
    ("CGo integration requires special handling", ["cgo"]),
    ("BTF generation for CO-RE", ["btf"]),
]


def _result_text(result: dict) -> str:
    """Extract the most relevant text from a result, regardless of type."""
    return (
        result.get("session_summary")
        or result.get("text")
        or result.get("text_preview", "")
    )


def measure_rag_cost(searcher: Searcher, question: str, top_k: int = 5) -> dict:
    """Query RAG and measure token cost of returned context."""
    enc = tiktoken.get_encoding("cl100k_base")
    results = searcher.query(question, top_k=top_k)

    per_result = []
    session_count = 0
    merged_count = 0

    for r in results:
        text = _result_text(r)
        tokens = len(enc.encode(text))
        entry = {
            "source": r["source"],
            "path": r["path"],
            "section": r.get("section", ""),
            "score": r["score"],
            "tokens": tokens,
            "chars": len(text),
        }
        if r["source"] == "sessions":
            session_count += 1
            entry["type"] = "session"
            entry["has_summary"] = "session_summary" in r
            entry["matched_turns"] = len(r.get("matched_turns", []))
        elif r.get("merged_chunks"):
            merged_count += 1
            entry["type"] = "merged"
            entry["merged_chunks"] = r["merged_chunks"]
        else:
            entry["type"] = "chunk"

        if r.get("doc_context"):
            entry["has_doc_context"] = True

        per_result.append(entry)

    total_tokens = sum(c["tokens"] for c in per_result)
    total_chars = sum(c["chars"] for c in per_result)
    return {
        "question": question,
        "top_k": top_k,
        "results": per_result,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "session_results": session_count,
        "merged_results": merged_count,
    }


def print_cost_report(cost: dict) -> None:
    """Print per-result and total token costs."""
    print(f"\n{'='*70}")
    print(f"RAG Cost Analysis (top_k={cost['top_k']})")
    print(f"{'='*70}")
    print(f"Question: {cost['question']}\n")

    print(f"{'#':<3} {'Type':<8} {'Source':<10} {'Score':<7} {'Tokens':<8} {'Path'}")
    print(f"{'-'*3} {'-'*8} {'-'*10} {'-'*7} {'-'*8} {'-'*40}")
    for i, c in enumerate(cost["results"], 1):
        section = f" § {c['section']}" if c["section"] else ""
        extra = ""
        if c["type"] == "session" and c.get("has_summary"):
            extra = f" [summary, {c['matched_turns']} turns matched]"
        elif c["type"] == "merged":
            extra = f" [merged {c['merged_chunks']} chunks]"
        if c.get("has_doc_context"):
            extra += " [+preamble]"
        print(f"{i:<3} {c['type']:<8} {c['source']:<10} {c['score']:<7.4f} {c['tokens']:<8} {c['path']}{section}{extra}")

    print(f"\nTotal context: {cost['total_tokens']} tokens ({cost['total_chars']} chars)")
    print(f"  = ~{cost['total_tokens'] / 1000:.1f}K tokens added to Cursor's context window per query")
    print(f"  Sessions: {cost['session_results']}  Merged docs: {cost['merged_results']}  Individual chunks: {len(cost['results']) - cost['session_results'] - cost['merged_results']}")


def print_rubric_template(cost: dict) -> None:
    """Print a markdown template for manual A/B scoring."""
    sources = [c["path"] for c in cost["results"]]

    print(f"\n{'='*70}")
    print("Rubric Template (copy to markdown for manual scoring)")
    print(f"{'='*70}\n")

    print(f"## Benchmark: system-probe Bazel migration blockers\n")
    print(f"RAG context cost: **{cost['total_tokens']} tokens** ({len(cost['results'])} results)")
    print(f"Sources retrieved: {', '.join(sources)}\n")

    print("### Rubric\n")
    print("| # | Fact                                | No RAG | With RAG |")
    print("|---|-------------------------------------|--------|----------|")
    for i, (fact, _) in enumerate(RUBRIC, 1):
        print(f"| {i} | {fact:<35} |        |          |")
    print("|   | **Total**                           |   /8   |     /8   |")

    print("\n### Token usage (from Cursor session info)\n")
    print("|                          | No RAG | With RAG  | Delta     |")
    print("|--------------------------|--------|-----------|-----------|")
    print(f"|  RAG context tokens      |   0    | {cost['total_tokens']:>7}   | +{cost['total_tokens']:<7} |")
    print("|  Cursor tokens used      |   ?    |    ?      |    ?      |")
    print("|  Facts recalled          |  ?/8   |   ?/8     |   +?      |")
    print("|  Extra tokens per fact   |   -    |    ?      |           |")


def main():
    cache_dir = os.path.expanduser(
        os.environ.get("AGENTS_RAG_CACHE", "~/.cache/agents-rag")
    )

    searcher = Searcher(cache_dir)
    if not searcher.load():
        print("ERROR: No index found. Run reindex first.", file=sys.stderr)
        sys.exit(1)

    stats = searcher.source_stats()
    print(f"Index loaded: {searcher.chunk_count} chunks ({stats})")

    for top_k in [5, 10]:
        cost = measure_rag_cost(searcher, QUESTION, top_k=top_k)
        print_cost_report(cost)

    cost_5 = measure_rag_cost(searcher, QUESTION, top_k=5)
    print_rubric_template(cost_5)


if __name__ == "__main__":
    main()
