# Phase 2B.2 reference experiment

This small CPU-only run optimizes neural muscle weights. It is an example and
resume checkpoint, not a successful locomotion policy. Physics and bodies are
unchanged. The network is 30→16→8 (632 float64 parameters).

Configuration: optimizer seed 123, population 8, two elites, 20 generations,
mutation probability 0.2, Gaussian standard deviation 0.15, clipping [−5, 5].
Training seeds: 0, 17, 42. Simple ten-node body, 120 Hz, 0.1-second actions,
three-second passive settling, 12-second episodes, target 20 units right.
Every candidate uses every training seed, sequentially. No starvation or hidden
propulsion is used. Fitness is the unchanged Phase 2B.1 reward.

`checkpoint.json` contains the next population and RNG state after generation 20.
`metrics.json` contains the configuration and all completed candidate scores.
`best.json` is the final champion; `initial-best.json` is the initial population's
champion. `heldout.json` reports every policy/seed result on seeds 101–105,
which were excluded from optimization.

| Policy | Mean reward | Mean X displacement | Mean muscle energy | Targets reached |
| --- | --- | --- | --- | --- |
| Trained | −0.0368 | +2.014 | 5.727 | 0/5 |
| Initial champion | −0.1564 | +0.595 | 5.626 | 0/5 |
| Idle | −0.1193 | −0.010 | 0 | 0/5 |
| Random actions | −0.3443 | −0.291 | 13.407 | 0/5 |
| Scripted | +0.2577 | +8.904 | 7.088 | 0/5 |

All episodes were stable. Training reward improved from −0.1772 to +0.0481.
The final brain improved modestly over initialization on held-out seeds, but
scripted movement remains much stronger. No target was reached. This does not
establish sustained crawling, robust generalization, or a solved task.

Reproduce with Python 3.14.8 / NumPy 2.5.4 (Linux used for these measurements):

```powershell
.\.venv\Scripts\python.exe -m eidolon.evolve train --output runs\reproduction --generations 20 --population 8 --elites 2 --seed 123 --seeds 0 17 42 --duration 12 --target-distance 20 --hidden-layers 16
.\.venv\Scripts\python.exe -m eidolon.evolve evaluate --brain experiments\phase2b2\best.json --baseline-brain experiments\phase2b2\initial-best.json --seeds 101 102 103 104 105 --compare
.\.venv\Scripts\python.exe -m eidolon.evolve resume --checkpoint experiments\phase2b2\checkpoint.json --output runs\continuation --generations 5
.\.venv\Scripts\python.exe -m eidolon --brain experiments\phase2b2\best.json --training-run experiments\phase2b2 --view split --seed 101
```

`benchmarks.json` records Linux SDL-dummy frame timings and sequential generation
throughput: 1.40–2.32 ms mean frame time across four views at 1280×720, and 9.41
seconds for eight candidates × three seeds × 12-second episodes. Timing the next
generation did not modify these saved generation-20 artifacts.
`stress.json` records separate stability-only checks with the same weights:
seeds 101/102, 60-second episodes, target 400, both body modes. All four completed
stably. The detailed body and altered horizon are out of the training distribution;
these checks establish stability, not locomotion quality.

The graphical `--brain` command now runs continuously with normal survival biology.
Use `eidolon.evolve evaluate` to reproduce the timed comparisons above. The saved
brain, metrics, checkpoint, and historical benchmark data are unchanged.

Cross-platform bitwise equivalence and native Windows performance are unverified.
The episode API does not protect against every possible simulator exploit; this
reward and the short-horizon evaluation protocol remain experimental.
