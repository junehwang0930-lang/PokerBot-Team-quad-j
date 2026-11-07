# Experimental single-file guarded policies

These files preserve tested research candidates. At the user's request, the higher-sampling candidate is now the root bot for a consolidated ZIP download; the previous champion is backed up under research/previous-champion. This packaging decision does not establish stronger validation.

- Root `agent.py`: higher-sampling guarded policy, SHA256 `81832887c80955c075192f299bdca459289c00108d70ee56e2db349f7ce9206e`.
- `agent_fast.py` here: faster alternative, SHA256 `7642e9c912638e0ae0ae7406b2c1fdf9064a09278647f115c6f494b3a5b5a9cf`.

Either version works by replacing only the official bundle's agent.py. Both use its existing NumPy dependency, embed all policy data, and preserve server plumbing. The research files are not required at runtime. Requirements, game, evaluator, judge, Dockerfiles and official tests are unchanged from main.

The higher-sampling version used up to 7.11 of 8 seconds per half in measured restricted-container matches. The faster version's broad improvement remains inconclusive. A clean official Docker build is still unverified because the environment's certificate chain blocked the base-image dependency download. See RESULTS.md for measurements and limitations; its final repository-state statement describes the research round before this publication commit.
