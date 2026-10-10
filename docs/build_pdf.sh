#!/usr/bin/env bash
# Rebuild docs/technical_report.pdf after editing docs/technical_report.md (needs pandoc and xelatex). Windows: docs\build_pdf.bat
cd "$(dirname "$0")" && pandoc technical_report.md -o technical_report.pdf --pdf-engine=xelatex --lua-filter=apa_figures.lua \
  -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" && echo "written docs/technical_report.pdf"
