# Record every demo step for real (asciinema). Outputs in /tmp/demo/*.cast
set -e
cd "$(dirname "$0")"
mkdir -p /tmp/demo
for s in ${STEPS:-step1_data step2_q1search step3_q1test step4_q2 step5_q3 step6_q4 step7_system}; do
  echo "recording $s"; ~/.venvs/rec/bin/asciinema rec --overwrite -q --cols 118 --rows 30 -c "bash $s.sh" /tmp/demo/$s.cast
done
echo ALL RECORDED
