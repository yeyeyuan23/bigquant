# Transfer archives

Large competition data archives are not tracked by normal Git. Git tracks this
manifest plus helper scripts; the archive payloads should be distributed as
GitHub Release assets or another binary store.

## Current local archives

| Archive | Size bytes | SHA-256 |
|---|---:|---|
| `bigalpha_research_data_v3.tar.gz` | 1216600492 | `4fd914654a38f11fc2f6b0bd1153d9db32c927e6a157ffaeb26a9b11c407e912` |
| `factorlib_all36_u1000_2019_2023.tar.gz` | 362203436 | `e9f1139b3c4397c48a4e7270b525697f3d57c117fdf04164538f0025c09e0b58` |
| `bigalpha_research_data_v2.tar.gz` | 857452791 | `459bb593a33d817dd850a2e8465db01523a229e6c9c56886ff4aeaf6c4bc48bd` |
| `bigalpha_research_data_20260726.tar.gz` | 163436779 | `8ed3bc3c1de8ae13fd9253d992522bbfe1cf6caaf61dc4b4d8aebe7d03a6e46c` |

## Recommended handoff

Uploader:

```bash
gh auth refresh -h github.com
bash scripts/upload_transfer_release.sh data-transfer-20260728
```

Teammate:

```bash
bash scripts/download_transfer_release.sh data-transfer-20260728
```

The download script places assets under `data/transfers/` and verifies every
downloaded `.tar.gz` against `artifacts/transfers/SHA256SUMS`.
