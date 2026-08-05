# BO Forge Performance Benchmarks

These measurements are release evidence, not machine-independent CI thresholds.
Each value is the median of five warm-cache subprocess runs after one unrecorded
warm-up run on the release workstation.

## v2.4.1 Startup And qMFKG Hardening

Recorded on 2026-08-05 with Python 3.11.14 and BoTorch 0.17.2.

| Command | v2.4.0 median (s) | v2.4.1 median (s) | Change |
| --- | ---: | ---: | ---: |
| `bo_forge --version` | 2.2599 | 0.3595 | -84.1% |
| CLI help | 2.4678 | 0.3676 | -85.1% |
| Discrete qMFKG validation | 2.2384 | 0.3736 | -83.3% |
| Discrete qMFKG `q=1` suggestion | 4.3978 | 5.0533 | +14.9% |
| Discrete qMFKG `q=2` suggestion | 6.9443 | 6.6397 | -4.4% |
| Discrete qMFKG `q=4` suggestion | 14.6199 | 12.3885 | -15.3% |

The startup improvement comes from lazy public and CLI imports. qMFKG runtime
settings remain unchanged when the new controls are omitted, so suggestion
times should be interpreted as normal stochastic and machine-level variation.
