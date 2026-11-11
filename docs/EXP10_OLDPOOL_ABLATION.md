# Old-pool ablation (no first-fit in the references) -- per-dataset results

One dataset = one consecutive search; the value compared is the best cost of the whole search. 'old' = original repair pool (3 destroy x 3 repair), 'ff' = pool with first-fit insertion (Experiment 10's references). Both reference protocols are the original one: Kisialiou runs 5 restarts of 300 iterations, the adaptive ALNS (no improvement operators) runs for Kisialiou's search CPU time. The RL models were given the first-fit Kisialiou's time.

## 1. What first-fit did to the references

old/ff = geometric mean over datasets of (final cost with the old pool) / (final cost with first-fit): above 1.0000 means first-fit lowered the cost. 'ff cheaper' = datasets where the first-fit run is strictly cheaper. Fleet = mean number of vessels in the final solution. CPU old/ff = mean ratio of the two runs' search CPU seconds (the old pool defines its own budget).

| size | n | Kisialiou old/ff | ff cheaper | adaptive old/ff | ff cheaper | fleet K old / ff | fleet A old / ff | CPU old/ff |
|---|---|---|---|---|---|---|---|---|
| small | 50 | 1.1017 | 39/50 | 1.0978 | 50/50 | 2.70 / 2.40 | 2.62 / 2.40 | 0.924 |
| medium | 50 | 1.0726 | 48/50 | 1.0837 | 49/50 | 4.30 / 3.94 | 4.30 / 3.94 | 0.983 |
| large | 50 | 1.0829 | 50/50 | 1.0647 | 49/50 | 6.90 / 6.30 | 7.08 / 6.60 | 1.043 |

### 1b. The same at equal compute

Both runs cut to the smaller of the two budgets (from their restart rows; complete restarts only, so the longer run loses its cut restart -- Kisialiou restarts are long, so this handicaps the longer Kisialiou run noticeably).

| size | n | Kisialiou old/ff | adaptive old/ff |
|---|---|---|---|
| small | 50 | 1.1020 | 1.0978 |
| medium | 50 | 1.0644 | 1.0837 |
| large | 50 | 1.0776 | 1.0634 |

## 2. Does RL win again when the references lack first-fit? (ratio-to-best, 1.0000 = best)

Denominator in every row: best of {adaptive, Kisialiou, RL main} on the dataset, within the row's own experiment. Exp 8 = original instances, pool and wall-clock budget; Exp 10 = first-fit references on the new instances; Ablation = old-pool references on the new instances (RL main is the same run as in Experiment 10, given the first-fit budget); Ablation at equal compute = every method cut to the smaller of the old-pool and first-fit budgets (RL from its restart rows).

| size | experiment | n | adaptive | Kisialiou | RL main | best on dataset: A / K / RL |
|---|---|---|---|---|---|---|
| small | Exp 8 (old instances, old pool) | 50 | 1.0513 | 1.0364 | 1.0003 | 0 / 26 / 30 |
| small | Exp 10 (new instances, first-fit pool) | 50 | 1.0100 | 1.0052 | 1.0468 | 14 / 37 / 22 |
| small | **Ablation (new instances, old pool)** | 50 | 1.0899 | 1.0903 | 1.0162 | 1 / 16 / 45 |
| small | Ablation at equal compute | 50 | 1.0899 | 1.0906 | 1.0163 | 1 / 16 / 43 |
| medium | Exp 8 (old instances, old pool) | 50 | 1.1019 | 1.1025 | 1.0040 | 0 / 30 / 20 |
| medium | Exp 10 (new instances, first-fit pool) | 50 | 1.0269 | 1.0176 | 1.0223 | 0 / 46 / 4 |
| medium | **Ablation (new instances, old pool)** | 50 | 1.1021 | 1.0808 | 1.0089 | 0 / 27 / 23 |
| medium | Ablation at equal compute | 50 | 1.1021 | 1.0809 | 1.0088 | 0 / 27 / 23 |
| large | Exp 8 (old instances, old pool) | 50 | 1.0961 | 1.0799 | 1.0029 | 0 / 15 / 35 |
| large | Exp 10 (new instances, first-fit pool) | 50 | 1.0630 | 1.0035 | 1.0476 | 0 / 49 / 1 |
| large | **Ablation (new instances, old pool)** | 50 | 1.0929 | 1.0516 | 1.0107 | 0 / 28 / 22 |
| large | Ablation at equal compute | 50 | 1.0934 | 1.0517 | 1.0107 | 0 / 28 / 22 |

## 3. Paired comparison with Kisialiou, per dataset

RL/Kis = geometric mean over datasets of RL's final cost / Kisialiou's final cost (below 1.0000 = RL cheaper); 'RL cheaper' = datasets where RL is strictly cheaper. Fleet = RL's final vessel count compared with Kisialiou's: fewer / more datasets. Old pool: RL as evaluated vs old-pool Kisialiou; first-fit: Experiment 10's own comparison.

| size | n | RL/Kis (old pool) | RL cheaper | fleet RL fewer / more | RL/Kis (ff pool) | RL cheaper | fleet RL fewer / more |
|---|---|---|---|---|---|---|---|
| small | 50 | 0.9400 | 33/50 | 11 / 1 | 1.0357 | 10/50 | 0 / 5 |
| medium | 50 | 0.9370 | 23/50 | 20 / 1 | 1.0050 | 4/50 | 4 / 3 |
| large | 50 | 0.9624 | 22/50 | 22 / 2 | 1.0422 | 1/50 | 1 / 10 |

## 4. Every method, one denominator

Denominator: best of {RL main, RL ASt, adaptive old, Kisialiou old, adaptive ff, Kisialiou ff} on the dataset (only datasets where all six exist; each at the budget of its own protocol). Mean ratio-to-best, 1.0000 = best; in brackets the number of datasets on which the method has the lowest cost.

| size | n | RL main | RL ASt | adaptive old | Kisialiou old | adaptive ff | Kisialiou ff |
|---|---|---|---|---|---|---|---|
| small | 50 | 1.0470 (18) | 1.0302 (29) | 1.1211 (0) | 1.1212 (10) | 1.0103 (11) | 1.0054 (36) |
| medium | 50 | 1.0312 (1) | 1.0074 (6) | 1.1255 (0) | 1.1038 (2) | 1.0359 (0) | 1.0265 (43) |
| large | 50 | 1.0476 (0) | 1.0513 (1) | 1.1321 (0) | 1.0890 (0) | 1.0630 (0) | 1.0035 (49) |

## 5. By instance type (RL main / Kisialiou final cost, geometric mean; below 1.0000 = RL cheaper)

TW = share of installations with a time window; DF = demand/frequency level (DF1 many small calls ... DF5 few large calls). Cells: old pool / first-fit pool, n in brackets.

By time-window density:

| size | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| small | 1.034 / 1.071 (10) | 0.909 / 1.038 (10) | 0.952 / 1.001 (10) | 0.911 / 1.035 (10) | 0.901 / 1.034 (10) |
| medium | 0.931 / 0.994 (10) | 0.964 / 1.009 (10) | 0.926 / 0.984 (10) | 0.926 / 1.010 (10) | 0.939 / 1.029 (10) |
| large | 0.949 / 1.030 (10) | 0.928 / 0.998 (10) | 0.986 / 1.054 (10) | 0.954 / 1.070 (10) | 0.997 / 1.061 (10) |

By demand/frequency level:

| size | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| small | 0.924 / 1.001 (10) | 0.919 / 1.069 (10) | 0.917 / 1.000 (10) | 0.943 / 1.072 (10) | 1.001 / 1.038 (10) |
| medium | 0.920 / 0.990 (10) | 0.904 / 1.008 (10) | 0.929 / 0.987 (10) | 0.963 / 1.007 (10) | 0.971 / 1.034 (10) |
| large | 0.963 / 1.017 (10) | 0.908 / 1.016 (10) | 0.953 / 1.038 (10) | 0.968 / 1.024 (10) | 1.024 / 1.119 (10) |

