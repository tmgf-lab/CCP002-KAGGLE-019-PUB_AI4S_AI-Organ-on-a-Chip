source "$(dirname "$0")/common.sh"
say "The end-to-end system: bright-field images in, all four answers out (plate BR00117019: never used to fit any model)"
run "python vcp_system.py --plate BR00117019 --wells B16 B04 A01 --dmso 6 --chips apps/web_demo/examples/chips/230317_63_HPMEC_good.png apps/web_demo/examples/chips/230320_169_HPMEC_bad.png --out /tmp/demo/system"
