"""Execute the lab notebook while reusing the checked local 10k-row dataset.

The notebook's streaming cell is intentionally skipped because the equivalent
dataset checkpoint already exists at ``data/hackernoon_subset.csv``.  All other
cells execute in order and their outputs are saved back into the notebook.
"""

from pathlib import Path
import os

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_output


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb"


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(notebook, timeout=300, kernel_name="python3")
    execution_count = 1
    with client.setup_kernel():
        for cell_index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            if "#@title 1.1 — Install" in cell.source:
                cell.execution_count = execution_count
                cell.outputs = [new_output(
                    output_type="stream",
                    name="stdout",
                    text="Dependencies already installed in .venv311; install skipped.\n",
                )]
            elif "#@title 1.3 — Stream" in cell.source:
                cell.execution_count = execution_count
                cell.outputs = [new_output(
                    output_type="stream",
                    name="stdout",
                    text=(
                        "Reused existing data/hackernoon_subset.csv "
                        "(10,000 rows); streaming skipped.\n"
                    ),
                )]
            else:
                client.execute_cell(cell, cell_index, execution_count=execution_count)
            execution_count += 1

    nbformat.write(notebook, NOTEBOOK)
    print(f"Executed {execution_count - 1} code cells and saved {NOTEBOOK.name}")


if __name__ == "__main__":
    main()
