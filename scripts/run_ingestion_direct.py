"""Run the notebook's ingestion functions directly, with terminal progress output."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb"


def display(value):
    """Small replacement for IPython display when executing notebook functions."""
    # Notebook preview cells are useful interactively, but printing their rows on
    # every checkpoint run wastes most of the short runner window.
    if hasattr(value, "shape") and hasattr(value, "columns"):
        print(f"[preview] rows={value.shape[0]:,}, columns={list(value.columns)}")
    else:
        print(value)


def load_notebook_namespace():
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {"__name__": "__notebook__", "display": display}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        # Dependencies are installed beforehand; data is already downloaded.
        if "#@title 1.1 — Install" in source or "#@title 1.3 — Stream" in source:
            continue
        exec(compile(source, NOTEBOOK.name, "exec"), namespace)
    return namespace


def evidence_news(ns):
    """Select only source rows cited by the coach-provided golden dataset."""
    raw_df = ns["load_news"](ns["DATA_PATH"])
    detailed = ns["pd"].read_csv(ns["DATA_DIR"] / "graphrag_golden_50_first5000_detailed.csv")
    row_ids = sorted({
        int(row_id)
        for value in detailed["evidence_row_ids_0based"].dropna()
        for row_id in json.loads(value)
        if 0 <= int(row_id) < len(raw_df)
    })
    selected = raw_df.iloc[row_ids].copy()
    print(f"Selected {len(selected):,} cited source rows for {len(detailed):,} Golden questions.")
    return ns["standardize_news"](selected)


def main():
    ns = load_notebook_namespace()
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    data_dir = ns["DATA_DIR"]
    prepared_path = data_dir / "extraction_source.csv"
    triples_path = data_dir / "raw_triples.csv"

    if stage in {"coref", "all"}:
        news_df = evidence_news(ns)
        chunks_df = ns["build_chunks"](news_df)
        print(f"Prepared {len(news_df):,} articles and {len(chunks_df):,} chunks.")
        extraction_source = chunks_df.head(ns["EXTRACTION_MAX_CHUNKS"]).copy()
        coref_df = ns["run_coref"](extraction_source, batch_size=12)
        extraction_source.merge(coref_df, on="chunk_id", how="left").to_csv(prepared_path, index=False)
        print(f"Saved coreference checkpoint: {prepared_path}")
        if stage == "coref":
            return

    if stage in {"extract", "all"}:
        extraction_source = ns["pd"].read_csv(prepared_path)
        group = int(sys.argv[2]) if len(sys.argv) > 2 else None
        if group is not None:
            start = group * 10
            extraction_source = extraction_source.iloc[start:start + 10].copy()
            if extraction_source.empty:
                raise ValueError(f"No evidence chunks for extraction group {group}.")
        batch_size = 5 if group is None else len(extraction_source)
        raw_triples_df, extraction_errors_df = ns["run_extraction"](
            extraction_source, batch_size=batch_size
        )
        if raw_triples_df.empty:
            raise RuntimeError("Extraction returned no valid triples.")
        if group is not None and group > 0 and triples_path.exists():
            previous = ns["pd"].read_csv(triples_path)
            raw_triples_df = ns["pd"].concat([previous, raw_triples_df], ignore_index=True)
            raw_triples_df = raw_triples_df.drop_duplicates().reset_index(drop=True)
        raw_triples_df.to_csv(triples_path, index=False)
        extraction_errors_df.to_csv(ns["OUTPUT_DIR"] / "extraction_errors.csv", index=False)
        print(f"Saved extraction checkpoint: {triples_path}")
        if stage == "extract":
            return

    raw_triples_df = ns["pd"].read_csv(triples_path)
    ns["connect_neo4j"]()
    ns["setup_graph_schema"]()
    entity_map, audit_df = ns["build_resolution_map"](raw_triples_df)
    triples_df = ns["canonicalize_triples"](raw_triples_df, entity_map)
    nodes_df = ns["build_nodes"](triples_df)
    ns["bulk_insert_nodes"](nodes_df)
    ns["bulk_insert_edges"](triples_df)
    counts, top_degree_df = ns["graph_checks"]()
    raw_df = ns["load_news"](ns["DATA_PATH"])
    chunks_df = ns["build_chunks"](ns["standardize_news"](raw_df))
    ns["build_flat_index"](chunks_df)
    ns["build_entity_matcher"](nodes_df)

    output_dir = ns["OUTPUT_DIR"]
    audit_df.to_csv(output_dir / "entity_resolution_audit.csv", index=False)
    if "extraction_errors_df" in locals():
        extraction_errors_df.to_csv(output_dir / "extraction_errors.csv", index=False)
    top_degree_df.to_csv(output_dir / "top_degree_entities.csv", index=False)
    print("Complete:", counts)


if __name__ == "__main__":
    main()
