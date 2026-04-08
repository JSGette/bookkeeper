"""RAG context quality benchmark.

Measures the token cost of RAG-retrieved chunks (in Cursor/Claude tokens)
and outputs a rubric template for manual A/B comparison in Cursor.
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


def measure_rag_cost(searcher: Searcher, question: str, top_k: int = 5) -> dict:
    """Query RAG and measure token cost of returned context."""
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = searcher.query(question, top_k=top_k)

    per_chunk = []
    for c in chunks:
        text = c.get("text") or c.get("text_preview", "")
        tokens = len(enc.encode(text))
        per_chunk.append({
            "source": c["source"],
            "path": c["path"],
            "section": c.get("section", ""),
            "score": c["score"],
            "tokens": tokens,
            "chars": len(text),
        })

    total_tokens = sum(c["tokens"] for c in per_chunk)
    total_chars = sum(c["chars"] for c in per_chunk)
    return {
        "question": question,
        "top_k": top_k,
        "chunks": per_chunk,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
    }


def print_cost_report(cost: dict) -> None:
    """Print per-chunk and total token costs."""
    print(f"\n{'='*70}")
    print(f"RAG Cost Analysis (top_k={cost['top_k']})")
    print(f"{'='*70}")
    print(f"Question: {cost['question']}\n")

    print(f"{'#':<3} {'Source':<10} {'Score':<7} {'Tokens':<8} {'Path'}")
    print(f"{'-'*3} {'-'*10} {'-'*7} {'-'*8} {'-'*40}")
    for i, c in enumerate(cost["chunks"], 1):
        section = f" § {c['section']}" if c["section"] else ""
        print(f"{i:<3} {c['source']:<10} {c['score']:<7.4f} {c['tokens']:<8} {c['path']}{section}")

    print(f"\nTotal context: {cost['total_tokens']} tokens ({cost['total_chars']} chars)")
    print(f"  = ~{cost['total_tokens'] / 1000:.1f}K tokens added to Cursor's context window per query")


def print_rubric_template(cost: dict) -> None:
    """Print a markdown template for manual A/B scoring."""
    sources = [c["path"] for c in cost["chunks"]]

    print(f"\n{'='*70}")
    print("Rubric Template (copy to markdown for manual scoring)")
    print(f"{'='*70}\n")

    print(f"## Benchmark: system-probe Bazel migration blockers\n")
    print(f"RAG context cost: **{cost['total_tokens']} tokens** ({len(cost['chunks'])} chunks)")
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

    # Use top_k=5 for the rubric template (default RAG query size)
    cost_5 = measure_rag_cost(searcher, QUESTION, top_k=5)
    print_rubric_template(cost_5)


if __name__ == "__main__":
    main()
