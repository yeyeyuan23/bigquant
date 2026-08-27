#!/usr/bin/env bash
set -euo pipefail

tag="${1:-data-transfer-20260728}"
title="${2:-BigAlpha research data transfer ${tag}}"

assets=(
  "data/transfers/bigalpha_research_data_v3.tar.gz"
  "data/transfers/bigalpha_research_data_v3.tar.gz.sha256"
  "data/transfers/factorlib_all36_download_v1/factorlib_all36_u1000_2019_2023.tar.gz"
  "data/transfers/factorlib_all36_download_v1/factorlib_all36_u1000_2019_2023.tar.gz.sha256"
)

missing=()
for asset in "${assets[@]}"; do
  if [[ ! -f "${asset}" ]]; then
    missing+=("${asset}")
  fi
done

if (( ${#missing[@]} > 0 )); then
  printf 'missing transfer assets:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 2
fi

if ! gh release view "${tag}" >/dev/null 2>&1; then
  gh release create "${tag}" --title "${title}" --notes "BigAlpha transfer archives. Verify with artifacts/transfers/SHA256SUMS."
fi

gh release upload "${tag}" "${assets[@]}" --clobber
