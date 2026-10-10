# Demo steps for the video: each command is typed, then really run (recorded by record_all.sh). They expect the run
# outputs used in the video: runs/winner weights, results/, the Q3 virtual-stain cache and the Q4 embeddings.
# Shared by every step: a typed prompt, then the command really runs. Nothing below is shown in the recording.
export PATH="$HOME/.venvs/vcp/bin:$PATH" PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VCP_WORKERS=4
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1  # repository root
say() { printf '\e[38;5;245m# %s\e[0m\n' "$1"; sleep 1.2; }
run() {
  printf '\e[1;38;5;208mvcp\e[0m \e[1m$\e[0m '
  local s="$1"; for ((i = 0; i < ${#s}; i++)); do printf '%s' "${s:i:1}"; sleep 0.022; done
  printf '\n'; sleep 0.4
  # eval in this shell (so `export` carries over to the next command); filter only library warning noise
  eval "$1" > >(grep --line-buffered -v -E "Warning|warn\(|^\s*$|oneDNN|cpu_feature|rebuild TensorFlow|^\[I 20") 2>&1
  sleep 1.5
}
export TERM=xterm-256color; clear
