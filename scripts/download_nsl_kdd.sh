#!/usr/bin/env bash
# Fetch NSL-KDD (Tavallaee et al., 2009) into data/raw/.
# The canonical home is https://www.unb.ca/cic/datasets/nsl.html (a download form, no stable direct link).
# This script tries public mirrors and VALIDATES what it got (43 columns, plausible row count); if nothing
# validates it tells you how to place the files manually. It never fabricates data.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

MIRRORS=(
  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master"
  "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master"
)

validate() {  # $1 file, $2 min rows
  [ -s "$1" ] || return 1
  local cols rows
  cols=$(head -n1 "$1" | awk -F',' '{print NF}')
  rows=$(wc -l < "$1")
  [ "$cols" -eq 43 ] && [ "$rows" -ge "$2" ]
}

fetch() {  # $1 filename, $2 min rows
  local name="$1" min="$2" url enc
  enc="${name//+/%2B}"
  validate "data/raw/$name" "$min" && { echo "[ok] data/raw/$name already present and valid"; return 0; }
  for base in "${MIRRORS[@]}"; do
    url="$base/$enc"
    echo "[..] trying $url"
    if curl -fsSL --max-time 120 "$url" -o "data/raw/$name.tmp" && validate "data/raw/$name.tmp" "$min"; then
      mv "data/raw/$name.tmp" "data/raw/$name"; echo "[ok] $name <- $base"; return 0
    fi
    rm -f "data/raw/$name.tmp"
  done
  return 1
}

ok=1
fetch "KDDTrain+.txt" 100000 || ok=0
fetch "KDDTest+.txt"  20000  || ok=0

if [ "$ok" -eq 0 ]; then
  cat <<MSG

[!!] Automatic download failed (mirrors may have moved, or you are offline).
     Get the files yourself and place them at:
         data/raw/KDDTrain+.txt
         data/raw/KDDTest+.txt
     Sources:  https://www.unb.ca/cic/datasets/nsl.html
               kaggle datasets download -d hassan06/nslkdd   (needs a Kaggle account/API token)
     Expected: comma-separated, no header, 43 columns (41 features + label + difficulty).
MSG
  exit 1
fi
echo "Train rows: $(wc -l < 'data/raw/KDDTrain+.txt')   Test rows: $(wc -l < 'data/raw/KDDTest+.txt')"
