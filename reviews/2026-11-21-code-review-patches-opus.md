# Code review: llama.cpp patches 0003-0006 (and 0001 against upstream 690b87d)

- Date: 2026-11-21
- Reviewer: Claude Opus 5.5, independent of the earlier GPT-6.2 Sol review (`reviews/2026-11-21-code-slot-patch-gpt-6.2-sol.md`)
- Scope: reading and git only. Nothing was built or run. Claims marked "inference" come from reading the code, not from a run.
- Tree: `vendor/llama.cpp`, branch `inkling-patches`, HEAD `9f5b9d6b7`, clean. The `git patch-id --stable` of each of the 7 commits on 29b52d3..HEAD matches the patch file of the same number. The vendor tree is the patch series.

## Guide for Daniel

**0003: a slot may carry its own routed experts.** Read `src/models/inkling2-loop.cpp:64-108` and `src/llama-model.cpp:2009-2012`.
- `load_arch_tensors` still copies each physical layer into its slots (line 66). Then, for each slot, it creates `blk.<slot>.ffn_gate_up_exps.weight` and `blk.<slot>.ffn_down_exps.weight` only when the file has them. It records which ones it made in `slot_own_gate_up` and `slot_own_down` (`models.h:2558-2559`).
- The generic scale pass then runs over all layers, slots included. A slot that owns a projection gets the `.scale` of its own name, if the file has one.
- `share_slots()` (lines 95-108) replaces the 0001 loop. It copies the physical layer into the slot again and puts back the slot's own weight and scale for each projection it owns.

**0004 and 0006: the test** `--loop-slots`. Read `tests/test-llama-archs.cpp:67-84` (`fill_key`) and `:2002-2056` (`test_loop_slots`). The fixture has 3 physical layers and core [1,2). 0006 sets the pass count to 3, so slot 3 is pass 2 and slot 4 is pass 3. Listed slot tensors are placeholders. `fill_key` can fill a slot tensor with the values of `blk.1`. There are 9 checks: 2 no-ops must give exactly the same logits, and 7 perturbations must change them.

**0005: find slot experts in every split.** Read `inkling2-loop.cpp:74-77`. The presence check now uses the loader's `weights_map`, which indexes every split. It falls back to the GGUF metadata only for a metadata-only model (`ml.files.empty()`). 0005 also fixes the slot example in the comment: pass 2 over layer 6 is slot 24.

**0001 against upstream.** Upstream 690b87d (#32704) fixes the same bug in a different way. A virtual `post_arch_tensors` hook runs after the scale pass, and the slot copy moves out of `load_arch_tensors` into that hook. On a base at or after 690b87d, 0001 is redundant. 0003 then has to be rewritten by hand (finding 1). On the current pin, 29b52d3, 690b87d is not present (it is 11 commits later), so 0001's fix is still needed. 0003 already replaced 0001's loop with `share_slots()`.

**What I checked and found correct:**
- Slot mapping. The graph uses `il = n_phys + (pass-1)*(end-start) + (ip-start)` with pass 0-based (`inkling2-loop.cpp:139`). For 1-based p this gives `24 + (p-2)*12 + (l-6)`. `layer_phys` (line 4) inverts it. `scripts/perpass.sh:23` uses the same formula.
- No-op path. When a file has no slot tensors, `listed()` only does a `weights_map.find` with no side effects, nothing is created, and `share_slots()` reduces to the old loop. The slot layers hold the same pointers as before, so the graph is byte-identical. The real-file check agrees: `results/pilot/perpass-parity.txt` says "pp-noop == pp-base and pp-allslots == pp-all, byte-identical".
- Scales. Inkling2 requires fused `gate_up` (`inkling2.cpp:154`, flag 0), so the gate_up scale lives in `ffn_up_exps_s`, and `share_slots` keeps the right pair. A slot with its own weight and no own scale runs unscaled, which is correct for a donor without scales. `build_moe_ffn` gets no input scales (`inkling2.cpp:396-412`), so `*_in_s` has no effect on any path.
- Memory and ownership. `llama_layer` and its nested structs hold only raw `ggml_tensor *` pointers. The loader's contexts own the tensors, and copying a layer copies pointers only. Nothing is freed twice. Each listed slot tensor is created exactly once, with a shape check against the file.
- Split lookup. `weights_map` is filled from every split (`llama-model-loader.cpp:737`) and from the file-pointer path (`:797`).

## Findings

1. **Major. The series still applies cleanly onto 690b87d, but 0003 lands in the wrong function.** I applied 0001-0007 with `git apply --cached` to a scratch index read from 690b87d, and all seven applied without a conflict. `git merge-tree 690b87d inkling-patches` also reports no conflict. In the result, 0003's slot-creation block sits inside `post_arch_tensors(llama_model_loader &)`. That parameter has no name, so `ml.` resolves to the member `llama_model_loader * ml` (`llama-model.h:1088`). Inference: the build fails at compile time. If someone then just names the parameter, the code compiles but breaks: `share_slots()` (still called at `llama-model.cpp:2011`) indexes `slot_own_*` before they are sized, and the slot tensors are created after the scale pass, so their scales are never attached. **Fix when moving the pin:** drop 0001. Delete the early copy. In `load_arch_tensors`, create the owned slot experts on otherwise empty slot layers, let the generic pass attach their scales, and call `share_slots()` from `post_arch_tensors`. With this order the scale pass probes only names the slot owns. An orphan slot scale (Sol #3) would then fail loudly at the tensor count instead of being dropped. For the pilot, keep 29b52d3. The range 29b52d3..690b87d also changes `src/llama-graph.cpp` and `src/llama-memory-hybrid-idx.cpp` (`git diff --stat`), so moving the pin could shift numbers already measured (inference).

2. **Minor. 0005 has no test that fails without it.** The fixture is metadata-only. Its loader constructor fills no `weights_map` (`llama-model-loader.cpp:800-806`, `n_tensors = weights_map.size()`), so `get_tensor_meta` always returns null and the old `gguf_find_tensor` fallback does all the work. Inference: 0006 passes with or without 0005. This breaks the CLAUDE.md rule "Each patch ships with a test in `test-llama-archs` that fails without it". The real runs do not cover splits either: no file under `models/` has an `-of-` split name. Impact on results: none, because every pilot file is a single file.

3. **Minor. The tensor-count guard never runs in the tests.** In metadata-only mode, `create_tensor` returns a synthesized tensor without counting it (`llama-model-loader.cpp:1387-1428`), so `done_getting_tensors` is vacuous. A file that has a slot tensor the code never creates only fails on a real file. Examples are a split (finding 2), an orphan scale, or a slot beyond P. This matters for the P = 4 rerun (inference from `load_arch_hparams`, lines 24-30):
   - At P = 4 there are 36 slots, and slots 48-59 alias the physical experts. In a per-pass file, pass 4 therefore runs the base experts. A per-pass condition at P = 4 needs `slots DONOR 4` added.
   - A file with pass-3 slots (36-47) fails to load at P = 2, and the error is loud.
   - A file with no slot tensors, such as `cand.gguf`, is not affected.

4. **Minor. The scale check doubles both projections' slot scales at once.** The `scaled` case calls `apply_expert_rescale` on every `blk.4.*_exps.scale`. Inference: drop either `layers[il].ffn_up_exps_s = own.ffn_up_exps_s` (line 101) or `layers[il].ffn_down_exps_s = own.ffn_down_exps_s` (line 105) and the other projection still changes the logits, so the test passes. Split it into a gate_up-only check and a down-only check.

5. **Minor. The tests do not pin which pass reads which slot, or the layer offset.** The fixture core is one layer wide (`LOOP_START 1`, `LOOP_END 2`, `test-llama-archs.cpp:767-768`). So `(l-6)` and the stride of 12 are never exercised, and a pass-2/pass-3 swap would pass all 9 checks: both slots still change the logits and still differ from each other. The real-file parity runs do not catch a swap either, because they put the same donor in both passes. Today the mapping rests on reading upstream `inkling2-loop.cpp:139`, which is correct. The H1 contrast between p2 and p3 depends on it. **Test:** run with a `cb_eval` that records `src[0]->name` of each `MUL_MAT_ID` by pass. Use a fixture with a core 2 wide and 3 passes, and assert that pass 2 over core layer k reads `blk.<n_phys + k>` and pass 3 reads `blk.<n_phys + w + k>`.

6. **Minor. The positive controls are close to zero.** `results/env.md` records nmse 6.163e-10 to 2.573e-09 for the seven perturbation checks, against a threshold of `> 0.0`. They pass at seed 42 on CPU, which is deterministic. Sol reproduced false failures at `-d 0.001`. The noise is not the issue. The issue is that the gap to zero is only a few ulps of logit change. Nothing to fix for the pilot. Do not lower `stdev` in this test.

7. **Nit. Orphan slot scales are still dropped silently (Sol #3, still open).** In the current order, a slot that does not own a projection still has that projection's aliased weight during the scale pass. The pass therefore creates a slot-named `.scale` if one exists, and `share_slots` then discards it. `scripts/splice.py:25` copies only `.weight` and the matching `.scale` together, so our files cannot contain an orphan. The rewrite in finding 1 fixes this as a side effect.

8. **Nit. `splice.py --slot` does not check that PHYS belongs to SLOT.** Lines 70-80 accept any pair. A wrong pair puts the experts in another layer's pass. Expert shapes match across the core, so `--verify` passes. `perpass.sh` computes the pairs correctly. A one-line guard would close this: `PHYS == core_start + (SLOT - n_phys) % (core_end - core_start)`, from the file's metadata.

9. **Nit. Stale docs.** CLAUDE.md says the build "must print three OK lines", but `results/env.md` now lists 12 (3 from loop-scales, 9 from loop-slots). It also credits `--loop-slots` to "patch 0003/0004", but the current test is 0006.

10. **Nit. `n_slot` means two different things.** In `load_arch_hparams` it counts all layers (line 24). In `load_arch_tensors` it counts cache slots only (line 71).

## Verdict

The patches 0003-0006 are correct for the pilot's files:
- single-file GGUFs,
- slots built by `splice.py` with both projections,
- no MoE cache, LoRA, imatrix collection on slot files, or pruning.

I found no bug that changes a pilot result. Before the H1 result depends on pass identity, add the test in finding 5. Before moving the llama.cpp pin, do the rewrite in finding 1. 0001 is redundant only on a base at or after 690b87d.
