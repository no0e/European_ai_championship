# Task 3 — results

| Submitted file | Public score | Time (22/11) | Note |
|---|---|---|---|
| `plat6.pt` | **0.4700011591814447** | 07:59 | average of ps1, ps10, st1, bt1, bt2, y1 — **best** |
| `st1_a1.0.pt` | 0.4646 | 07:19 | best single model |
| `ps1_a1.0.pt` | 0.4605 | 06:06 | |
| `bt2_a1.0.pt` | 0.4597 | 06:49 | |
| `y1_a1.0.pt` | 0.4596 | 07:40 | |
| `bt1_a1.0.pt` | 0.4591 | 06:39 | |
| `ps10_a1.0.pt` | not submitted | | seed 1 of `ps1` |
| `plat10.pt` (12-model average) | 0.4690 | 08:22 | |

The submitted `plat6.pt` (a ViT-B/16 `state_dict`, 343,617,075 B, sha1 `978e7668096ee8affd463ac1b6b2ac58da87ff8e`)
is far above the Git file size limit, so it is attached to the GitHub release
[`submitted-files`](../../../releases/tag/submitted-files). Steps 03 and 04 rebuild it from the six averaged models
(about 20 min on a 4-GPU node).

Full history (92 team submissions, local statistics of each candidate): `task3/history/HISTORY.md` on branch `task3-k3`.
