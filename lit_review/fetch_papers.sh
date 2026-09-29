#!/usr/bin/env bash
# Downloads arXiv LaTeX sources into ./<folder>/src and the PDF into ./<folder>/paper.pdf
# Run from this folder:  bash fetch_papers.sh
# Delete lines from the PAPERS list for any paper you don't want.
set -u
cd "$(dirname "$0")"

PAPERS="
core/01_rcGAN_Bendel2022 2210.13389
core/02_MMGAN_Whitney2024 2410.24197
core/03_RCPS_Bates2021 2101.02703
core/04_Im2ImRCPS_Angelopoulos2022 2202.05265
core/05_AIFS-CRPS_Lang2024 2412.15832
suggested/06_ScoringRuleGenNets_Pacchiardi2021 2112.08217
suggested/07_CramerGAN_Bellemare2017 1705.10743
suggested/08_WGAN-GP_Gulrajani2017 1704.00028
suggested/09_pcaGAN_Bendel2024 2411.00605
suggested/10_ConformalIntro_Angelopoulos2021 2107.07511
"

UA="Mozilla/5.0 (lit-review fetch; academic use)"
echo "$PAPERS" | while read -r dir id; do
  [ -z "${dir:-}" ] && continue
  echo "== $dir ($id)"
  mkdir -p "$dir/src"
  curl -sSL -A "$UA" -o "$dir/eprint.bin" "https://arxiv.org/e-print/$id" || { echo "  download failed"; continue; }
  kind=$(file -b "$dir/eprint.bin")
  case "$kind" in
    *gzip*)
      if tar -tzf "$dir/eprint.bin" >/dev/null 2>&1; then
        tar -xzf "$dir/eprint.bin" -C "$dir/src"
      else
        gunzip -c "$dir/eprint.bin" > "$dir/src/main.tex"
      fi ;;
    *tar*) tar -xf "$dir/eprint.bin" -C "$dir/src" ;;
    *PDF*) mv "$dir/eprint.bin" "$dir/paper.pdf"; echo "  NO LATEX AVAILABLE (PDF only)" ;;
    *) mv "$dir/eprint.bin" "$dir/src/main.tex" ;;
  esac
  rm -f "$dir/eprint.bin"
  [ -f "$dir/paper.pdf" ] || curl -sSL -A "$UA" -o "$dir/paper.pdf" "https://arxiv.org/pdf/$id"
  echo "$id" > "$dir/ARXIV_ID"
  sleep 3   # be polite to arXiv
done

# Not on arXiv (no LaTeX exists publicly): Gneiting & Raftery (2007), author-hosted PDF
mkdir -p suggested/11_ProperScoringRules_Gneiting2007
curl -sSL -A "$UA" -o suggested/11_ProperScoringRules_Gneiting2007/paper.pdf \
  "https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf" \
  || echo "Gneiting2007 PDF failed - download manually"

# Ferro (2014) is paywalled (Wiley/QJRMS): download via UCL access into this folder
mkdir -p suggested/12_FairScores_Ferro2014
[ -f suggested/12_FairScores_Ferro2014/paper.pdf ] || echo "REMINDER: save Ferro 2014 PDF as suggested/12_FairScores_Ferro2014/paper.pdf (doi:10.1002/qj.2270)"

echo "Done. Tell Claude to finish the index."
