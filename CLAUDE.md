# CLAUDE.md — Project instructions for AI coding agents

## What this project is
A quantisation faithfulness study. See PROTOCOL.md for research questions.

## Hard rules
- Never add CUDA, PyTorch, bitsandbytes, AutoGPTQ, AWQ or vLLM as dependencies
- Never add argparse, click, typer or any CLI framework
- Never add async/await, threading or multiprocessing
- Never add Flask, FastAPI, Streamlit, Gradio or any web framework
- Never add Docker, CI/CD or GitHub Actions configuration
- Never branch eval logic on precision tier — precision differs only in model file path
- All generation calls go through the Backend protocol in backends/types.py
- All raw results are JSONL, all summaries are CSV
- All configs are YAML loaded with yaml.safe_load()
- All paths use pathlib.Path
- All functions have type hints
- Google-style docstrings, brief
- Line length 99 (ruff)
- Tests run with: pytest tests/

## Key files
- configs/model_primary.yaml — frozen model files and SHA256 hashes
- PROTOCOL.md — pre-registered statistical methods (do not change after Phase 2)
- results/raw/run_manifest.jsonl — append-only run log

## Running
Scripts are in scripts/ and run in order: 00 through 06.
Each script loads its config from configs/ and writes to results/raw/.

## Additional project notes

These supplement the rules above and do not override them.

- PROTOCOL.md section 10 lists every deviation from the original build
  specification, with reasons. Add to that table rather than changing the
  protocol silently.
- The refusal pattern list in eval/refusal_classifier.py is frozen. Changing it
  invalidates every previously scored behavioural result.
- Do not add stub modules. A file listed in the spec's directory tree is created
  when it is implemented, not before.
- A result file without a matching run_manifest.jsonl entry is not evidence.
  Every suite run appends exactly one manifest line.
