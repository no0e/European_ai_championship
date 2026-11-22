# Task 1 — results

| Submission | Public score | Time | CV (5 folds) | Content |
|---|---|---|---|---|
| no. 1081 | 0.4531 | 21/11 16:50 | 0.4621 | R1: 8×8 decoder map + threshold per method |
| no. 1155 | 0.4890 | 21/11 18:10 | — | R3: R1 + 75 TrustMark rectangles |
| no. 1660 | 0.594 | 21/11 ~21:30 | — | dense detector v1 (label bug) |
| no. 1694 | 0.713 | 21/11 21:54 | 0.6765 | dense detector v2 |
| no. 1820 | 0.7381 | 21/11 | 0.7085 | v3 (CV fine-tuning) |
| no. 1883 | 0.7463 | 21/11 23:10 | 0.7162 | v3 + guided filter |
| no. 2170 | 0.7469 | 22/11 00:47 | 0.7213 | v3 + v4 + guided filter |
| no. 3173 | 0.7467 | 22/11 06:10 | 0.7431 (biased) | + second-scale maps without folds |
| no. 3491 | 0.7555 | 22/11 07:36 | 0.7298 | m1_sel: first frame models |
| no. 3530 | 0.7714 | 22/11 07:47 | 0.7599 | m2b: + second-scale models |
| no. 3562 | 0.7754 | 22/11 07:57 | 0.7632 | m3: + second-scale bases with frame |
| **no. 3606** | **0.7872614154749301** | **22/11 08:13** | **0.7723** | **m5 (`sub_m5.pt`, sha1 317a0ead1e) — best** |

The submitted `sub_m5.pt` (about 16 MB, sha1 `317a0ead1e…`) is **not in this repository**: it was written on the
scratch of the JURECA project (`training2666`), whose access closed when the event ended, and it was never copied off
the cluster. `bash run_all.sh` rebuilds an equivalent file in `out/sub_m5.pt` (m5 minus v7/v7p/v8 — see the task
README, section 7). The sha1 is kept here so the submitted file can still be identified if a copy turns up.

`submissions.csv`: raw journal written by `submit_file.py` from 21/11 22:48 on (sha1 of each file, notes, HTTP status).
