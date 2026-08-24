# Validation status

Validated on 2026-08-18 in the ChatGPT execution environment:

- `PYTHONPATH=src pytest -q` -> **6 passed**.
- `PYTHONPATH=src bash scripts/smoke.sh` -> **PASS** end-to-end with the mock backend.
- The smoke path exercised baseline execution, counterfactual collection, world-model training, motif mining, paired macro admission, DarwinReasoner execution, resume-safe result storage, and paper-table aggregation.

Not validated in this environment:

- A100/CUDA execution;
- actual vLLM model loading;
- the real benchmark datasets named in example configs;
- official third-party SOTA baseline integrations;
- distributed Slurm behavior on the target cluster.

Those must be checked on the borrowed GPU server with `bash scripts/check_server.sh` followed by a tiny real-model pilot before any long sweep.
