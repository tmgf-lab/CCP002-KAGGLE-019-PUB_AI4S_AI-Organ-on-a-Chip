source "$(dirname "$0")/common.sh"
say "Q2: guided search for one source x label: virtual-stain features, non-toxic drugs vs DMSO (40 paths, 8 workers)"
run "VCP_VERBOSE_PATHS=1 python -m sweep.q2_search search --source virtual --label treated_vs_dmso --budget 40 --workers 8 --hours 0.1 --features runs/q1_test/site_features.csv --out /tmp/demo/q2"
run "python show.py search /tmp/demo/q2/searches/q2_virtual_treated_vs_dmso_summary.json"
