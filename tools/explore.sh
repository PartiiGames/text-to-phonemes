#!/bin/sh
# Farm-only exploration: espeak-ng version/API, Piper's phonemizer source, the voice's phoneme_id_map, CMUdict.
# Writes out/ (artifacts). espeak-ng is a test-time comparison tool only; never a dependency of the crate.
set -u
cd "$(dirname "$0")/.." && mkdir -p out && cd out
echo "== espeak"; command -v espeak-ng; espeak-ng --version
ls /usr/lib/x86_64-linux-gnu/ | grep -i espeak
nm -D /usr/lib/x86_64-linux-gnu/libespeak-ng.so.1 2>/dev/null | grep -i TextToPhonemes
WORDS="hello world water butter better city party winter twenty ladder button kitten mountain little about again america sofa wanted roses remember decide because happy react being into to the a an and of for more door north four car air there near sure fire hour her were hurry early bird information international national computer beautiful phoneme dog lot thought father judge church sing yes red go day boy my now book boot seat cat"
echo "== piper"
python3.12 -m venv /tmp/pv && /tmp/pv/bin/pip -q install piper-tts 2>&1 | tail -3
/tmp/pv/bin/pip -q download --no-deps -d wheels piper-tts 2>&1 | tail -2; ls wheels; /tmp/pv/bin/python -m zipfile -l wheels/piper*.whl | tail -40
/tmp/pv/bin/python -c "import piper,os;print(os.path.dirname(piper.__file__))" > piperdir.txt && D=$(cat piperdir.txt) && ls "$D" && for f in phonemize_espeak.py phoneme_ids.py voice.py config.py; do echo "=== $f"; cat "$D/$f"; done
ls "$D"/*.so 2>/dev/null
WORDS="$WORDS" /tmp/pv/bin/python - <<'PY'
try:
    from piper.phonemize_espeak import EspeakPhonemizer
    import piper, os
    p = EspeakPhonemizer()
    for w in os.environ["WORDS"].split(): print(w, repr(p.phonemize("en-us", w)))
    for t in ["Hello, world. How are you today? I'm fine; thanks: really! The cat sat on the mat.", "It was 1990 and I paid $3.50 for 25% of it at 3:30 pm on the 2nd. Dr. Smith lives on Elm St.", "water"]:
        print(repr(p.phonemize("en-us", t)))
except Exception as e:
    import traceback; traceback.print_exc()
PY
echo "== voice"
for n in hifi_tts_92 hfc_female lessac; do
  curl -fsSL -o "en_US-$n-medium.onnx.json" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/$n/medium/en_US-$n-medium.onnx.json" && echo "got $n"
done
curl -fsSL https://huggingface.co/api/models/rhasspy/piper-voices/tree/main/en/en_US | python3.12 -c "import json,sys;print([x['path'] for x in json.load(sys.stdin)])"
for f in *.onnx.json; do python3.12 -c "import json,sys;c=json.load(open(sys.argv[1]));print(sys.argv[1],c.get('espeak'),c.get('phoneme_type'),len(c['phoneme_id_map']));print(''.join(sorted(c['phoneme_id_map'])))" "$f"; done
echo "== cmudict"
curl -fsSL https://api.github.com/repos/cmusphinx/cmudict/commits/master | python3.12 -c "import json,sys;print(json.load(sys.stdin)['sha'])" > cmudict.sha
cat cmudict.sha; S=$(cat cmudict.sha)
curl -fsSL -o cmudict.dict "https://raw.githubusercontent.com/cmusphinx/cmudict/$S/cmudict.dict"
curl -fsSL -o cmudict.LICENSE "https://raw.githubusercontent.com/cmusphinx/cmudict/$S/LICENSE"
wc -l cmudict.dict; sha256sum cmudict.dict cmudict.LICENSE; head -3 cmudict.dict; cat cmudict.LICENSE
curl -fsSL -o words.txt https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa-no-swears.txt; wc -l words.txt
echo done
