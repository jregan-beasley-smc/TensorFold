# PR608 Qwen CUDA first-token validation

The runtime fix commits and hands off the actual first token before preparing initial DFlash2
proposals. It is commit `7acdc54473b5c9f54582b5d3c465e89b92daf886` above PR608 head
`99bcc5b29805e29a4baf2cd4beade0f7fe1a5fe8`. These validation files are separate from that commit.
CUDA kernels, checkpoint precision, weights and sampling arithmetic are unchanged.

## Environment and scope

DGX Spark, NVIDIA GB10 (`sm_121`), Linux aarch64, Zig 0.17.0, native CUDA.
Target: `Vontra/Qwen3.8-27B-MLX-4bit`, revision `70ae7fac63274ff2eac54152031433374cb80f2f`.
Drafter: `z-lab/Qwen3.8-27B-DFlash2`, revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
Both variants used the same 49-kernel Triton AOT pack and native CUDA assets. The pack was captured
in NVIDIA's PyTorch 26.07 container; exact binary/source/AOT hashes are in the JSON. The public bundle does not
independently retain CUDA driver/library versions or distribute the compiled binaries.

Fresh processes ran **A/B/A/B**, each cell with one first-use burst and three warm bursts.
Normal pair A tested greedy coding at 1/4/16 streams and sampled chat at four; pair B repeated
coding at 1/16. The control repeated coding at 1/16 only. Server parallelism was 16, context 32768,
segments 1, prompt cache 0, drafts on and thinking off; every request generated 128 tokens.
Coding uses a 14-token plain completion prompt. Chat is templated; actual prompt counts are in
its receipt. Requests within a cell share one prompt, with seed 1234 for greedy and 1234+i for
sampled stream i. This is short synthetic serving, not a heterogeneous workload or quality test.

## Observations

Seconds below are the median of three **warm burst metrics**. Within a burst, median TTFT is the
request median and slowest TTFT is its maximum. Batch wall spans earliest send to last completion.
All first-use observations, request timings and fingerprints remain in the JSON.
The upstream gap used slowest TTFT, so median improvements are not directly comparable to it.

### Normal shipping binaries

| Cell/pair | Median TTFT base → fix | Slowest TTFT base → fix | Batch wall base → fix |
|---|---:|---:|---:|
| code n1 T0 / A | 0.13442 → 0.12125 | 0.13442 → 0.12125 | 2.07753 → 2.22653 |
| code n4 T0 / A | 0.36522 → 0.36805 | 0.36555 → 0.37176 | 2.94397 → 2.96931 |
| code n16 T0 / A | 0.63523 → 0.59785 | 0.63633 → 0.62035 | 5.69084 → 5.73026 |
| chat n4 T1 / A | 0.38349 → 0.36246 | 0.38391 → 0.37346 | 4.30768 → 4.35212 |
| code n1 T0 / B | 0.12767 → 0.12195 | 0.12767 → 0.12195 | 2.08484 → 2.08272 |
| code n16 T0 / B | 0.63017 → 0.58046 | 0.63127 → 0.60285 | 5.70427 → 5.72430 |

### Separate calibration-control binaries

| Cell/pair | Median TTFT base → fix | Slowest TTFT base → fix | Batch wall base → fix |
|---|---:|---:|---:|
| code n1 T0 / A | 0.12693 → 0.12079 | 0.12693 → 0.12079 | 2.07244 → 2.08129 |
| code n16 T0 / A | 0.63724 → 0.59844 | 0.63856 → 0.62234 | 5.74129 → 5.72921 |
| code n1 T0 / B | 0.12685 → 0.12021 | 0.12685 → 0.12021 | 2.08155 → 2.08252 |
| code n16 T0 / B | 0.64489 → 0.59218 | 0.64721 → 0.61653 | 5.73058 → 5.75734 |

Normal n1 warm pair A had about 7.2% longer full wall: 26 verification rounds versus 24 in the
baseline. Pair B used 24 in both arms and had flat wall. The control retained normal calibration
and warmup in each process, then reused the first baseline's measured 14-entry host planning-cost
table. Its n1 arms used 24 rounds; median batch wall was within 0.5% in the four control pairs.
This supports a startup-calibration confound, not general throughput equivalence. Online adaptation
remained enabled; two n16 baseline warm bursts performed more speculative work than the candidate.
Four-stream coding was effectively flat, sampled chat variable, and first-use observations mixed.

Earlier first emission lengthens the first-to-last decode interval. Client decode TPS can fall
without a later completion. We retain unadjusted full response/batch wall, without subtracting time.
The result is a bounded warm first-token benefit, not a throughput gain or closure of the full
Python-versus-Zig gap. Two same-order pairs do not establish a confidence interval.

## Correctness and retained evidence

`normal.json` retains 372 checked requests and `calibration_control.json` retains 288: 660 total.
All completed with 128 output tokens, cache 0 and matching prompt-token work. Drafted/plain solo,
concurrent/solo and cross-variant output fingerprints matched. Evidence is a 12-hex server token
fingerprint plus a separate full streamed content/reasoning SHA256, **not retained literal token IDs**.
Host tests separately cover terminal EOS/max1, cancellation, forced first, logprobs, exactly-once
delivery and cleanup after initial-draft failure.

Independent host controls passed 2/2; the original failed six desired edge cases and the fix passed 8/8.
Core tests passed 70/70, focused host tests 7/7, and Linux CUDA-host suites 284/284 base and 287/287 fix.
The affected 141 tests passed again after native-wrapper capability forwarding. These observations
qualify neither every family nor long contexts/model quality. Deployment/restore records are excluded.
Binary/cubin hashes in this bundle are guarded collection attestations; readers cannot rehash absent
binaries from these JSON files alone.

## Reproduce

Use Python 3.11+, Zig 0.17.0, a qualified CUDA build and the pinned model revisions. Follow
`CONTRIBUTING.md` and `packaging/README.md` for builds/assets. Model downloads and AOT capture are
prerequisites, not actions performed by this client. Run one isolated server at a time. Set
`TARGET_DIR`, `DRAFTER_DIR` and `AOT_DIR` to your own matching files:

```sh
TENSORFOLD_CUDA_KERNELS="$AOT_DIR" TENSORFOLD_MEMORY_RESERVE_GIB=8 \
  ./zig-out/native/bin/tensorfold-native serve "$TARGET_DIR" \
  --backend cuda --drafter "$DRAFTER_DIR" --host 127.0.0.1 --port 8080 \
  --name qwen27-test --parallel 16 --context 32768 --segments 1 \
  --prompt-cache-gib 0 --temperature 0 --no-thinking --no-update-check --dashboard
```

From a separate terminal:

```sh
python3 validation/pr608-first-token/burst.py http://127.0.0.1:8080 qwen27-test \
  --tools tools --prompt code --streams 16 --temperature 0 \
  --tokens 128 --reps 3 --label base-a-code-16-t0 --output base-a-code-16-t0.jsonl
```

Restart in base-a, candidate-a, base-b, candidate-b order. In pair A, run code n1/T0, code n4/T0,
code n16/T0, then chat n4/T1; in pair B run code n1/T0 then n16/T0. Change the client options/label
accordingly and use separate output files. First-use is the first burst in that cell invocation,
not necessarily the first work of the process. The client refuses output overwrite and retains
all bursts plus drafted/plain solo references. It delegates bodies and timing to unchanged
`tools/bench_concurrent.py`, recording client/helper hashes; it does not manage servers or services.

For the control, apply `calibration-control.patch` to **separate diagnostic copies** of each revision
and rebuild. The patch must not enter the shipping fix. Start base-a with
`TF_PR608_CALIBRATION_LOG=1` and `TF_PR608_FIXED_COSTS` unset. Read its single `PR608_CALIBRATION`
JSON line and pass the ordered `ms` values as comma-separated `TF_PR608_FIXED_COSTS` to the other
three processes, retaining logging. All four still perform normal calibration/warmup before the
optional override. Check every logged width/value equals the first table; do not substitute the
recorded table as if it were your machine's calibration. This is not a proposed scheduler policy.

Require complete JSONL terminal records and equal request multisets of
`(seed, prompt_tokens, tokens, token_sha, output_sha256)` for matching cells/solos across variants.
Retain per-request server `rounds`/`accepted`/`drafted` as well: identical output work does not imply
identical speculative compute. Report raw TTFT and full wall, not character-estimated or adjusted TPS.
