source "$(dirname "$0")/common.sh"
say "Q1: the full-scale winner (U-Net w64, 2 seeds + test-time augmentation) on the unseen test plates, 30 fields each"
run "python evaluate.py --winner runs/q1_final/q1_winner_local.json --out /tmp/demo/q1test --limit 30 --workers 8 > /tmp/demo/q1test.log"
run "python show.py q1test /tmp/demo/q1test"
