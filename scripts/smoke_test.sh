#!/usr/bin/env bash
# Offline end-to-end smoke test on the synthetic fixture corpus. CPU only, no network.
#   default            : doctor + FTS build + ask + denylist + remote-gate + eval
#   --with-mock        : also exercise openai_compatible LLM+embedding providers via
#                        tests/mock_openai_server.py on 127.0.0.1 (no real model)
#   --with-vectors     : also run sentence_transformers vectors (DOWNLOADS a model; opt-in)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-python3}"; [[ -x .venv/bin/python ]] && PY=.venv/bin/python
MOCK=0; VEC=0
for a in "$@"; do case "$a" in --with-mock) MOCK=1;; --with-vectors) VEC=1;; esac; done
OUT=tests/.out; rm -rf "$OUT"; mkdir -p "$OUT/index"
CFG=tests/fixtures/config.smoke.json
export LOCAL_DOC_KB_CONFIG="$ROOT/$CFG"
fail=0; pass() { echo "PASS $*"; }; bad() { echo "FAIL $*"; fail=1; }; skip() { echo "SKIP $*"; }

$PY scripts/doctor.py --config $CFG --no-network >"$OUT/doctor.txt"; [[ $? == 0 ]] && pass doctor || { cat "$OUT/doctor.txt"; bad doctor; }
if command -v swiftc >/dev/null; then
  tools/macos-vision-ocr/build.sh >"$OUT/vision.txt" 2>&1 && grep -q "^PASS " "$OUT/vision.txt" && pass "visionocr self-test" || { cat "$OUT/vision.txt"; bad visionocr; }
else skip "visionocr (no swiftc)"; fi
$PY scripts/build_fts.py --config $CFG >"$OUT/fts.txt" 2>&1 && pass "build_fts: $(cat $OUT/fts.txt)" || bad build_fts
$PY scripts/build_fts.py --config $CFG | grep -q "skipped_unchanged=5" && pass "incremental rebuild skips unchanged" || bad incremental
$PY scripts/ask_hybrid_v2.py "coffee water temperature" --config $CFG -n 3 >"$OUT/ask1.txt" 2>&1
grep -q "^## 1\. notes/coffee_brewing.md" "$OUT/ask1.txt" && pass "ask top1 coffee" || { cat "$OUT/ask1.txt"; bad ask1; }
grep -q "excerpt:.*92-96" "$OUT/ask1.txt" && pass "excerpt is around the query, not the file start" || { cat "$OUT/ask1.txt"; bad excerpt; }
$PY scripts/ask_hybrid_v2.py "coffee water temperature" --config $CFG --json -n 3 >"$OUT/ask1.json"
$PY -c 'import json,sys; d=json.load(open(sys.argv[1])); h=d["hits"][0]; assert h["relpath"].endswith("coffee_brewing.md"); assert "92-96" in h["excerpt"]' "$OUT/ask1.json" && pass "ask --json evidence pack" || { cat "$OUT/ask1.json"; bad ask-json; }
grep -q "channels off: vectors(disabled), desc(disabled), media(disabled), ocr(disabled)" "$OUT/ask1.txt" && pass "graceful degradation message" || bad degrade-msg
$PY scripts/ask_hybrid_v2.py "番茄 日照" --config $CFG -n 3 | grep -q "^## 1\. notes/zh_garden.md" && pass "ask Chinese query" || bad ask-zh
if command -v pdftotext >/dev/null; then
  $PY scripts/ask_hybrid_v2.py "kayak buoyancy" --config $CFG -n 3 | grep -q "kayak_safety.pdf" && pass "pdf extracted+found" || bad pdf
else skip "pdf (no poppler)"; fi
$PY scripts/ask_hybrid_v2.py "DENYLIST_CANARY_7731" --config $CFG -n 5 | grep -q "secret_should" && bad denylist || pass "denylist excludes private/"
$PY scripts/eval_queries.py tests/fixtures/queries.jsonl --config $CFG -k 3 >"$OUT/eval.txt"; tail -1 "$OUT/eval.txt" | grep -q "hit@1=1.00" && pass "eval $(tail -1 $OUT/eval.txt)" || { cat "$OUT/eval.txt"; bad eval; }

# remote gate: saas embedding without opt-in must exit 3
RC="$OUT/remote.json"
$PY - "$CFG" "$RC" <<'PY'
import json,sys; c=json.load(open(sys.argv[1])); c["features"]["vectors"]=True
c["providers"]={"embedding":{"type":"saas","model":"x","base_url":"https://api.example.com/v1","api_key_env":"FAKE_KEY"}}
json.dump(c,open(sys.argv[2],"w"))
PY
$PY scripts/build_vectors.py --config "$RC" >/dev/null 2>"$OUT/gate.txt"; rc=$?
[[ $rc == 3 ]] && grep -q "SECURITY WARNING" "$OUT/gate.txt" && pass "remote gate refuses (allow_remote=false)" || bad "remote gate rc=$rc"
$PY -c "import json,sys;c=json.load(open('$RC'));c['allow_remote']=True;json.dump(c,open('$RC','w'))"
$PY scripts/build_vectors.py --config "$RC" >/dev/null 2>&1; rc=$?
[[ $rc == 3 ]] && pass "remote gate refuses without --i-understand-remote" || bad "gate2 rc=$rc"

if [[ $MOCK == 1 ]]; then
  PORT=18765; $PY tests/mock_openai_server.py $PORT & MPID=$!; trap 'kill $MPID 2>/dev/null' EXIT; sleep 1
  MC="$OUT/mock.json"
  $PY - "$CFG" "$MC" $PORT <<'PY'
import json,sys; c=json.load(open(sys.argv[1])); p=sys.argv[3]
c["features"].update(vectors=True, desc=True)
c["build"]={"desc_types":[".md",".txt",".pdf"]}
base=f"http://127.0.0.1:{p}/v1"
c["providers"]={"llm":{"type":"openai_compatible","base_url":base,"model":"mock-llm"},
                "embedding":{"type":"openai_compatible","base_url":base,"model":"mock-embed"}}
json.dump(c,open(sys.argv[2],"w"))
PY
  $PY scripts/doctor.py --config "$MC" | grep -Eq "^PASS +provider llm" && pass "doctor provider health (mock)" || bad doctor-mock
  $PY scripts/build_desc.py --config "$MC" --conc 2 >"$OUT/desc.txt" 2>&1 && grep -q "DONE" "$OUT/desc.txt" && pass "build_desc via openai_compatible" || { cat "$OUT/desc.txt"; bad desc; }
  $PY scripts/build_vectors.py --config "$MC" >"$OUT/vec.txt" 2>&1 && pass "build_vectors via openai_compatible ($(grep -o 'dim=[0-9]*' $OUT/vec.txt|head -1))" || { cat "$OUT/vec.txt"; bad vec; }
  $PY scripts/build_desc_vectors.py --config "$MC" >"$OUT/dvec.txt" 2>&1 && pass "build_desc_vectors" || { cat "$OUT/dvec.txt"; bad dvec; }
  $PY scripts/ask_hybrid_v2.py "chain lubricate" --config "$MC" -n 3 >"$OUT/ask_mock.txt" 2>&1
  grep -q "vector [1-9]" "$OUT/ask_mock.txt" && grep -q "bicycle" "$OUT/ask_mock.txt" && pass "ask with vector+desc channels" || { cat "$OUT/ask_mock.txt"; bad ask-mock; }
  # with a vector index present, a remote query embedder must also be gated
  $PY scripts/ask_hybrid_v2.py "coffee" --config "$RC" >/dev/null 2>&1; rc=$?
  [[ $rc == 3 ]] && pass "ask also gated" || bad "ask gate rc=$rc"
  # model mismatch: change embedding model id -> builder refuses (4), ask --strict-model refuses (4)
  $PY -c "import json;c=json.load(open('$MC'));c['providers']['embedding']['model']='other-embed';json.dump(c,open('$OUT/mm.json','w'))"
  $PY scripts/build_vectors.py --config "$OUT/mm.json" >/dev/null 2>&1; [[ $? == 4 ]] && pass "builder refuses mixed embedding models" || bad mm-build
  $PY scripts/ask_hybrid_v2.py "chain" --config "$OUT/mm.json" --strict-model >/dev/null 2>&1; [[ $? == 4 ]] && pass "ask refuses model mismatch (--strict-model)" || bad mm-ask
  $PY scripts/ask_hybrid_v2.py "chain" --config "$MC" --body-model other-embed --strict-model >/dev/null 2>"$OUT/arm.txt"; rc=$?
  [[ $rc == 4 ]] && grep -q "embedding mismatch" "$OUT/arm.txt" && pass "ask alternate arm checks index_meta" || { cat "$OUT/arm.txt"; bad "arm rc=$rc"; }
  $PY scripts/ask_hybrid_v2.py "chain" --config "$OUT/mm.json" 2>&1 | grep -q "embedding mismatch" && pass "ask warns + falls back on mismatch" || bad mm-warn
  $PY scripts/doctor.py --config "$OUT/mm.json" | grep -q "FAIL  index vectors" && pass "doctor flags index/model mismatch" || bad mm-doctor
else skip "provider pipeline (pass --with-mock)"; fi

if [[ $VEC == 1 ]]; then
  VC="$OUT/st.json"
  $PY -c "import json;c=json.load(open('$CFG'));c['features']['vectors']=True;c.setdefault('index',{})['vectors']='tests/.out/index/vectors_st.sqlite3';c['providers']={'embedding':{'type':'sentence_transformers','model':'${SMOKE_ST_MODEL:-BAAI/bge-small-zh-v1.5}','device':'cpu'}};json.dump(c,open('$VC','w'))"
  $PY scripts/build_vectors.py --config "$VC" && $PY scripts/ask_hybrid_v2.py "brew coffee" --config "$VC" -n 3 | grep -q "vector [1-9]" && pass sentence_transformers || bad st
else skip "sentence_transformers vectors (pass --with-vectors; downloads a model)"; fi

echo; [[ $fail == 0 ]] && echo "SMOKE OK" || echo "SMOKE FAILED"; exit $fail
