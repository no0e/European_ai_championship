# Task 3 — results

| Submitted file | Public score | Time (22/11) | Note |
|---|---|---|---|
| `plat6.pt` | **0.470** | 07:59 | average of ps1, ps10, st1, bt1, bt2, y1 — **best** |
| `st1_a1.0.pt` | 0.465 | 07:19 | best single model |
| `ps1_a1.0.pt` | 0.461 | 06:06 | |
| `bt2_a1.0.pt` | 0.460 | 06:49 | |
| `y1_a1.0.pt` | 0.460 | 07:40 | |
| `bt1_a1.0.pt` | 0.459 | 06:39 | |
| `ps10_a1.0.pt` | not submitted | | seed 1 of `ps1` |
| `plat10.pt` (12-model average) | 0.469 | 08:22 | |

The submitted `plat6.pt` (a ViT-B/16 `state_dict`, 343,617,075 B, sha1 `978e7668096ee8affd463ac1b6b2ac58da87ff8e`)
is far above the Git file size limit, so it is attached to this repository's GitHub release `submitted-files`.
Steps 03 and 04 rebuild it from the six averaged models (about 20 min on a 4-GPU node).

We made 92 submissions on this task during the event; the table above keeps the ones that matter — the submitted
file, the best single model, every model that enters the average, and the 12-model average we tried afterwards. The
raw submission log is not part of this repository.
