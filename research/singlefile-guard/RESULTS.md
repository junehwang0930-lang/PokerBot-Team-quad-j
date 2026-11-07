# Single-file research round

## Decision
Retain the existing champion. Neither new guarded configuration is promoted. The high-sample port has promising limited-population evidence but little runtime margin. The faster port's population improvement is inconclusive. This round does not complete the broader world-class-AI research mission, establish a Nash approximation, or demonstrate superiority to Jish.

## Implemented code
- candidate.py embeds the original Jish Monte Carlo preflop estimates, with vectorized combinadic indexing, into the champion. It requires no external table or extra package. Table indexing is exact; the equity estimates themselves are Monte Carlo estimates, not exact enumerated equities.
- numpy_guard.py ports the guarded Jish policy to the champion's NumPy evaluator. It embeds the table and champion fallback policy. Opponent samples exclude visible cards; each runout excludes the sampled opponent's hand. Learning uses revealed cards only. The existing stale-cache guard is retained.
- numpy_fast.py halves the initial sample counts, reduces the minimum from 96 to 48, and uses two rather than three pre-river runouts.
- Both guarded ports preserve server plumbing byte-for-byte and depend only on the official bundle plus its existing NumPy dependency. No Numba or auxiliary data files are required.

## New measurements
Table transfer: two 1000-deal head matches averaged -0.00575 bb/duplicate deal against champion. Against actual Jish, its paired improvement over champion averaged +0.18675 across two development seeds. Not sufficient for promotion.

Each guarded configuration received 24 full duplicate comparison matches: two independent seed blocks, six opponents, both candidate and champion on identical deals. These are 48000 individual hands per configuration. The confidence intervals below use the two block averages and Student t with one degree of freedom, not individual hands as independent replicates. These are nominal exploratory intervals, not multiplicity-adjusted or robust guarantees. Two blocks are a very small sample.

| Opponent | High-sample port | Paired champion | Faster port | Paired champion |
|---|---:|---:|---:|---:|
| Jish | +0.2670 | -2.8108 | +0.1310 | -3.4538 |
| Pressure | +0.9400 | +0.4943 | +0.4900 | +0.9988 |
| Trapper | +2.9068 | +2.5440 | +2.5508 | +3.1720 |
| Always pot | +8.4213 | +8.5348 | +8.2998 | +8.6848 |
| Calling station | +11.9703 | +11.3185 | +11.8440 | +11.5078 |
| Random | +10.2340 | +9.1513 | +10.5053 | +9.8003 |

All values bb/duplicate deal. Configurations used different fresh seeds; their absolute scores cannot be compared directly.

- High-sample paired field gain +0.917875, nominal 95% CI [0.517100,1.318650].
- Faster paired field gain +0.518500, nominal 95% CI [-0.467290,1.504290].
- High-sample development head vs champion +2.1255; vs Jish +0.0320. These are separate development results.

## Compatibility and timing
Both ports passed 12 official tests plus three research tests checking unchanged server plumbing, import dependencies, sample-card legality, and memory reset. The port evaluator matched the official evaluator on 2000 randomly generated complete deals. All 376992 preflop indices were verified, with sampled scalar/vector parity.

Restricted full-dependency container: 1 CPU, 4 GB, non-root, no network, read-only filesystem, writable tmpfs. High-sample startup 1.8 seconds; faster startup 2.1 seconds.

- High-sample vs champion: +2.5315 bb/deal, judge CI approximately [1.711,3.352], max thinking 6.15/8 seconds.
- High-sample vs calling station: +13.097 bb/deal, max thinking 7.11/8 seconds.
- Faster vs calling station: +12.971 bb/deal, max thinking 6.65/8 seconds.
- These three valid HTTP matches had zero errors. See JSON artifacts for full fault statistics. An initial station run had an unreadable opponent mount and forfeited; that invalid result is excluded and preserved separately.
- Direct harness maximum half times: high-sample 7.2555 seconds, faster 4.6659 seconds. Direct times exclude HTTP and hand-end costs and cannot substitute for judge timing.

Unmodified official check_build.sh was rerun and failed in the base-image pip install with SSL certificate verification errors. The runtime image was an existing reconstructed full-dependency image. A clean official build remains unverified.

## Training and method scope
This round reused existing trained parameters, the prior 226150-decision dataset, and original Jish preflop labels. No new neural network or CFR policy was trained. New games generated validation rewards rather than action-imitation targets. The research scripts, source hashes, per-deal rewards, timing logs, and summaries are preserved here.

Deep CFR, DeepStack and NFSP were reviewed at the paper-abstract level (arxiv.org/abs/1811.00164, arxiv.org/abs/1701.01724, arxiv.org/abs/1603.01121). Their principles motivate sampled regret learning and value approximation, but their reported Hold'em results do not establish performance in this game. No Libratus/Pluribus comparison, exploitability bound, approximate-equilibrium opponent, large new self-play training run, or comprehensive literature study was completed in this round.

## Remaining weaknesses
1. Runtime margin: high-sample implementation reaches 7.11 seconds against a passive opponent; hardware variation matters.
2. Strategic robustness: the faster policy loses value against pressure and trapping opponents. Near-even Jish results do not demonstrate a winning edge against Jish.
3. Validation coverage: only two fresh blocks and six styles per configuration, mostly related synthetic opponents; no reliable estimate against the unknown tournament field or a certified equilibrium strategy.

The root agent.py and remote branches were not changed during this round. The previously pushed Numba-based challenger remains a separate earlier experiment.
