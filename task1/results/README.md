# Task 1 — results

| Submission | Public score | Time | CV (5 folds) | Content |
|---|---|---|---|---|
| no. 1081 | 0.453 | 21/11 16:50 | 0.462 | R1: 8×8 decoder map + threshold per method |
| no. 1155 | 0.489 | 21/11 18:10 | — | R3: R1 + 75 TrustMark rectangles |
| no. 1660 | 0.594 | 21/11 ~21:30 | — | dense detector v1 (label bug) |
| no. 1694 | 0.713 | 21/11 21:54 | 0.677 | dense detector v2 |
| no. 1820 | 0.738 | 21/11 | 0.709 | v3 (CV fine-tuning) |
| no. 1883 | 0.746 | 21/11 23:10 | 0.716 | v3 + guided filter |
| no. 2170 | 0.747 | 22/11 00:47 | 0.721 | v3 + v4 + guided filter |
| no. 3173 | 0.747 | 22/11 06:10 | 0.743 (biased) | + second-scale maps without folds |
| no. 3491 | 0.756 | 22/11 07:36 | 0.730 | m1_sel: first frame models |
| no. 3530 | 0.771 | 22/11 07:47 | 0.760 | m2b: + second-scale models |
| no. 3562 | 0.775 | 22/11 07:57 | 0.763 | m3: + second-scale bases with frame |
| **no. 3606** | **0.787** | **22/11 08:13** | **0.772** | **m5 (`sub_m5.pt`, sha1 317a0ead1e) — best** |

The submitted `sub_m5.pt` is attached to this repository's GitHub release `submitted-files`: 16,734,551 B, sha1
`317a0ead1eee7df22bb1867ae68526c0a04d2ff7`. It holds 75 TrustMark rectangles and 945 dense maps,
no constant prediction, median predicted fraction 0.46 (from the assembly log). `bash run_all.sh` rebuilds an
equivalent file in `out/sub_m5.pt`, close but not identical — see the task README, section 7.

`submissions.csv`: raw journal written by `submit_file.py` from 21/11 22:48 on (sha1 of each file, notes, HTTP status).
