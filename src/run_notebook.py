"""
run_notebook.py
===============
Executes 05_multimodal_dual_encoder.ipynb via nbclient, saving outputs in place.
"""

import os
import sys
import time
import nbformat
from nbclient import NotebookClient

def main():
    if len(sys.argv) > 1:
        nb_path = sys.argv[1]
    else:
        nb_path = "NLP-Project/notebook/06_multimodal_dual_encoder_weighted_loss.ipynb"
    print(f"[*] Reading notebook from: {nb_path}")
    nb = nbformat.read(nb_path, as_version=4)

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": "NLP-Project/notebook"}}
    )
    
    start_time = time.time()
    print("[*] Starting notebook execution with CUDA / GPU acceleration...")
    client.execute()
    elapsed = time.time() - start_time
    print(f"[+] Execution completed in {elapsed:.2f} seconds!")

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[+] Saved executed notebook with outputs to: {nb_path}")

if __name__ == "__main__":
    main()
