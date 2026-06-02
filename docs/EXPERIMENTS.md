# Experiment Notes

Short planning log for RL/ALNS experiments. Group models by experiment number, then ID, so folder names stay comparable across runs.

| Experiment | ID | Action module | Idea | Compare against | Folder tag |
|---|---|---|---|---|---|
| 1 | 1 | `op_pair_v1` | Baseline pair selection | Original ALNS without improvement stage | `exp1_id1_op_pair_v1` |
| 1 | 2 | `op_pair_mapped_v1` | Same idea as ID 1 with mapped engine operators | ID 1 / Experiment 1 | `exp1_id2_op_pair_mapped_v1` |
| 2 | 3 | `improvement_chain_v1` | Kisialiou-style destroy/repair selection with RL improvement | Kisialiou ALNS performance | `exp2_id3_improvement_chain_v1` |
| 2 | 4 | `improvement_many_v1` | Same as ID 3, but with more operators | Kisialiou ALNS performance | `exp2_id4_improvement_many_v1` |
| 1 | 5 | `destroy_repair_many_v1` | Many destroy/repair operators, no improvement phase | Experiment 1 group | `exp1_id5_destroy_repair_many_v1` |