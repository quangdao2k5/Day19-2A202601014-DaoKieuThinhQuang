"""Produce the required evaluation and summary CSV artifacts from real evidence."""
from pathlib import Path
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent))
from run_ingestion_direct import load_notebook_namespace


def main():
    ns = load_notebook_namespace()
    pd = ns["pd"]
    cache_index = ns["OUTPUT_DIR"] / "flatrag.index"
    cache_store = ns["OUTPUT_DIR"] / "flatrag_chunks.parquet"
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "prepare" or not (cache_index.exists() and cache_store.exists()):
        raw = ns["load_news"](ns["DATA_PATH"])
        chunks = ns["build_chunks"](ns["standardize_news"](raw))
        ns["build_flat_index"](chunks)
        ns["faiss"].write_index(ns["flat_index"], str(cache_index))
        ns["flat_store"].to_parquet(cache_store, index=False)
        print(f"Cached FlatRAG index: {cache_index}")
    else:
        ns["flat_index"] = ns["faiss"].read_index(str(cache_index))
        ns["flat_store"] = pd.read_parquet(cache_store)
        print(f"Loaded cached FlatRAG index: {ns['flat_index'].ntotal} vectors")
    if mode == "prepare":
        return
    triples = pd.read_csv(ns["DATA_DIR"] / "raw_triples.csv")
    ns["connect_neo4j"]()
    nodes = ns["build_nodes"](ns["canonicalize_triples"](triples, ns["build_resolution_map"](triples)[0]))
    ns["build_entity_matcher"](nodes)
    golden = pd.read_csv(ns["GOLDEN_PATH"])
    ns["validate_golden"](golden, require_answers=True)
    question_index = int(mode) if mode != "all" else None
    target = golden.iloc[question_index:question_index + 1] if question_index is not None else golden
    results = ns["run_evaluation"](target)
    result_path = ns["OUTPUT_DIR"] / "graphrag_eval_results.csv"
    if question_index is not None:
        parts_dir = ns["OUTPUT_DIR"] / "eval_parts"
        parts_dir.mkdir(exist_ok=True)
        results.to_csv(parts_dir / f"{question_index:02d}.csv", index=False)
        part_paths = sorted(parts_dir.glob("*.csv"))
        results = pd.concat([pd.read_csv(path) for path in part_paths], ignore_index=True)
    elif result_path.exists():
        previous = pd.read_csv(result_path)
        results = pd.concat([previous, results], ignore_index=True).drop_duplicates("id", keep="last")
    summary = ns["comparison_table"](results)
    results.to_csv(result_path, index=False)
    summary.to_csv(ns["OUTPUT_DIR"] / "graphrag_vs_flatrag_summary.csv", index=False)
    print("Evaluation complete", len(results))


if __name__ == "__main__":
    main()
