# Included bot versions

The repository root `agent.py` is the higher-sampling, self-contained guarded policy. Run it with the root requirements and official bundle. All policy data is embedded; nothing under research is imported at runtime.

Other versions are backups, not components to combine:

- `singlefile-guard/agent_fast.py`: faster self-contained alternative. To use it, copy it over root agent.py.
- `previous-champion/agent.py`: prior validated champion. Copy over root agent.py to restore it.
- `numba-guard/agent.py`: older compiled challenger. To use it, copy its agent.py, requirements.txt and data directory into a separate official bundle. It needs Numba and data/preflop.npy.

The user requested consolidating the versions onto main for a single ZIP download. This publication is not evidence of additional strategic validation. The active policy has promising but limited population results and used up to 7.11 of 8 seconds per half in tested restricted-container matches. A clean official Docker build remains unverified due to the environment's certificate failure. See singlefile-guard/RESULTS.md for measured results and limitations.
