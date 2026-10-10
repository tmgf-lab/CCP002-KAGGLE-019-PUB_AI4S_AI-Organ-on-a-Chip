source "$(dirname "$0")/common.sh"
say "Data: stream 8 wells of an unseen CPJUMP1 plate from the public Cell Painting Gallery (S3, CC0)"
run "VCP_DATA=/tmp/demo/data python download.py --smoke --plates BR00117017 --sites 1"
run "ls -lh /tmp/demo/data | head -5"
