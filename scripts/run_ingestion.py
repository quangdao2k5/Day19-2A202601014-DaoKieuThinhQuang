"""Execute the local GraphRAG notebook through graph ingestion and index creation.

The original notebook is an instructional guide, so its expensive calls are
commented out.  This runner makes the same calls in a clean kernel and saves a
notebook with the resulting cell outputs for submission/audit.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb"
EXECUTED_NOTEBOOK = ROOT / "outputs" / "Day19_GraphRAG_executed_ingestion.ipynb"

DRIVER_CELL = """
connect_neo4j()
setup_graph_schema()

raw_df = load_news(DATA_PATH)
news_df = standardize_news(raw_df)
chunks_df = build_chunks(news_df)
if chunks_df.empty:
    raise RuntimeError("No chunks were produced from the downloaded dataset.")
print(f"Prepared {len(news_df):,} articles and {len(chunks_df):,} chunks.")

extraction_source = chunks_df.head(EXTRACTION_MAX_CHUNKS).copy()
coref_df = run_coref(extraction_source)
extraction_source = extraction_source.merge(coref_df, on="chunk_id", how="left")
raw_triples_df, extraction_errors_df = run_extraction(extraction_source)
if raw_triples_df.empty:
    raise RuntimeError("Extraction returned no valid triples; inspect Groq model/configuration.")

entity_map, entity_resolution_audit_df = build_resolution_map(raw_triples_df)
triples_df = canonicalize_triples(raw_triples_df, entity_map)
nodes_df = build_nodes(triples_df)
bulk_insert_nodes(nodes_df)
bulk_insert_edges(triples_df)
graph_counts, top_degree_df = graph_checks()

build_flat_index(chunks_df)
build_entity_matcher(nodes_df)

entity_resolution_audit_df.to_csv(OUTPUT_DIR / "entity_resolution_audit.csv", index=False)
extraction_errors_df.to_csv(OUTPUT_DIR / "extraction_errors.csv", index=False)
top_degree_df.to_csv(OUTPUT_DIR / "top_degree_entities.csv", index=False)
print("Ingestion and local indexes completed.")
"""


def main() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    # Dependencies are installed into .venv before this runner is invoked.
    nb.cells = [
        cell
        for cell in nb.cells
        if "#@title 1.1 — Install" not in "".join(cell.get("source", []))
    ]
    nb.cells.append(nbformat.v4.new_code_cell(DRIVER_CELL, metadata={"tags": ["executed-runner"]}))

    client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute(cwd=str(ROOT))
    EXECUTED_NOTEBOOK.parent.mkdir(exist_ok=True)
    nbformat.write(nb, EXECUTED_NOTEBOOK)
    print(f"Saved executed notebook: {EXECUTED_NOTEBOOK}")


if __name__ == "__main__":
    main()
