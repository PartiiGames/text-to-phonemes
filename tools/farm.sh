#!/bin/sh
# Farm job (Linux): build + test text-to-phonemes, (re)train the OOV model if asked, compare with espeak-ng via Piper.
#   sh platform/ai/core/text-to-phonemes/tools/farm.sh        (from the repo root; RETRAIN=1 retrains data/oov.model)
# Outputs in platform/ai/core/text-to-phonemes/out/. espeak-ng comes only from the piper-tts wheel, in a venv, for comparison.
# Caps: every step has a timeout; progress lines start with "== ".
set -eu
G=$(cd "$(dirname "$0")/.." && pwd); AI=$(cd "$G/../.." && pwd)
C=${K2_JOB_CACHE:-/tmp/g2p-cache}; T=${CARGO_TARGET_DIR:-$AI/target}
mkdir -p "$C" "$G/out" "$G/data"   # data/ is empty in git until cmudict.dict is committed
# progress: a line a minute naming the current step (each step still has its own hard timeout, the no-progress cap)
STEP=$G/out/.step; echo start > "$STEP"
( while sleep 60; do echo "== alive: $(cat "$STEP" 2>/dev/null) $(date +%T)"; done ) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
step() { echo "$1" > "$STEP"; echo "== $1"; }
CMU_SHA=74790861f652b15e4ac49015a90074ad62a27690
CMU_SUM=81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22
if [ ! -f "$G/data/cmudict.dict" ]; then
  echo "== fetch cmudict $CMU_SHA"
  curl -fsSL --max-time 120 -o "$G/data/cmudict.dict" "https://raw.githubusercontent.com/cmusphinx/cmudict/$CMU_SHA/cmudict.dict"
fi
echo "$CMU_SUM  $G/data/cmudict.dict" | sha256sum -c -
[ -f "$C/words.txt" ] || curl -fsSL --max-time 60 -o "$C/words.txt" https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa-no-swears.txt
V=en_US-hifi_tts_92-medium.onnx.json
[ -f "$C/$V" ] || curl -fsSL --max-time 60 -o "$C/$V" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hifi_tts_92/medium/$V"
if [ ! -x "$C/pv/bin/python" ]; then
  step "venv: pip install piper-tts==1.8.0 (comparison only)"
  python3.12 -m venv "$C/pv" && timeout 900 "$C/pv/bin/pip" install --progress-bar off piper-tts==1.8.0 2>&1 | grep -E "^(Collecting|Downloading|Successfully)" || true
fi

cd "$AI"
step "build"
timeout 1200 cargo build --release -p text-to-phonemes --bin g2p-tool 2>&1 | tail -5
if [ ! -f "$G/data/oov.model" ] || [ "${RETRAIN:-0}" = 1 ]; then
  step "train OOV model"
  timeout 1800 "$T/release/g2p-tool" train "$G/data/cmudict.dict" "$G/data/oov.model" "${MIN_COUNT:-2}" | tee "$G/out/train.txt"
  timeout 1200 cargo build --release -p text-to-phonemes --bin g2p-tool 2>&1 | tail -3
fi
step "test"
timeout 1200 cargo test --release -p text-to-phonemes 2>&1 | tail -60
step "stats"
timeout 120 "$T/release/g2p-tool" stats
ls -l "$T/release/g2p-tool" "$G/data/"
find "$T/release/build" -path '*text-to-phonemes*' -name lexicon.bin -exec ls -l {} \;
gzip -9c "$G/data/oov.model" | wc -c | sed 's/^/oov.model gzip bytes /'
cp "$G/data/oov.model" "$G/out/" 2>/dev/null || true
cp "$G/data/cmudict.dict" "$G/out/"
cp "$AI/Cargo.lock" "$G/out/Cargo.lock"
step "package"
timeout 600 cargo package -p text-to-phonemes --allow-dirty --no-verify 2>&1 | tail -5 || true
timeout 60 cargo package -p text-to-phonemes --allow-dirty --list > "$G/out/package-list.txt" 2>&1 || true
head -40 "$G/out/package-list.txt"
for f in "$T"/package/text-to-phonemes-*.crate; do [ -f "$f" ] && ls -l "$f" && cp "$f" "$G/out/"; done || true
step "compare"
timeout 1800 "$C/pv/bin/python" -I "$G/tools/compare.py" "$T/release/g2p-tool" "$C/words.txt" "${NWORDS:-5000}" \
  "$G/tests/sentences.txt" "$C/$V" "$G/out" > "$G/out/compare.txt" 2>&1 || echo "compare failed"
cat "$G/out/compare.txt"
echo "== done"
