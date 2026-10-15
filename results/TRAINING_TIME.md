# Training Time

Recorded training time for the two completed runs, including checkpoint saving. Model loading, downtime, failed attempts, and evaluation are excluded.

| Method | Training rounds | Total training time (hours) |
|---|---:|---:|
| GRPO | 20 | 14.78 |
| ObsLite + GRPO | 20 | 11.74 |

Observed total time reduction: **3.04 hours (20.56%)**.

## Time per Round

All values below are in minutes. Values are rounded to three decimal places; totals use the original measurements. A round is one training iteration, not one environment interaction.

| Round | GRPO (minutes) | ObsLite + GRPO (minutes) |
|---|---:|---:|
| 1 | 66.209 | 38.239 |
| 2 | 71.028 | 28.887 |
| 3 | 44.919 | 39.405 |
| 4 | 42.927 | 32.500 |
| 5 | 39.848 | 33.197 |
| 6 | 46.825 | 35.230 |
| 7 | 39.643 | 32.788 |
| 8 | 44.909 | 36.197 |
| 9 | 40.432 | 33.595 |
| 10 | 49.398 | 43.114 |
| 11 | 35.785 | 29.262 |
| 12 | 26.788 | 30.421 |
| 13 | 36.825 | 31.447 |
| 14 | 47.451 | 40.654 |
| 15 | 47.314 | 36.350 |
| 16 | 48.574 | 47.999 |
| 17 | 59.207 | 45.882 |
| 18 | 22.892 | 23.632 |
| 19 | 29.390 | 28.371 |
| 20 | 46.368 | 37.263 |
