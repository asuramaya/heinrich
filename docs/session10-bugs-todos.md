# Session 10: Bugs, TODOs, Avoided Problems, and Technical Debt

## Known Bugs

### Critical
1. **Composite capture routing broken** — `viewport="composite"` in MCP capture doesn't reach the browser's `captureComposite()`. The server sends the command but the browser falls through to regular vp0 capture. The `_handleCapture` check for `msg.viewport==='composite'` is correct in code but the browser may not have loaded the latest HTML. Needs: verify the poll delivers `viewport: "composite"` correctly, and that the composite uploads via `/api/capture-upload` signals the pending request.

2. **GIF capture timeout** — Remote GIF capture still times out for Qwen 0.5B (150K tokens). The `play=true` optimization helps but 78 frames (3 loops × 26 layers) with Fibonacci/turntable orbit is too slow. The MCP timeout is 300s but the browser's GIF encoding (LZW on 78 frames) can exceed this. Manual browser GIF capture works fine.

3. **Direction coloring race condition (partially fixed)** — The 50-PC fallback was removed and full-K server projection is used. But: if the server is slow (cold USB read ~4s), the user sees script colors for 4 seconds. No loading indicator in the cloud viewports during the fetch.

4. **Trajectory lines broken for extreme directions** — When pinning tokens that are far apart on PC0 (like Engel/sein in raw mode), trajectory lines extend way beyond the cloud because the per-layer normalization doesn't handle extreme outlier values. The lines go off-screen.

5. **`_fetchPredicts` shows "predicts..." forever on pseudo-layers** — Fixed for emb/lmh by checking `cL >= _nRealLayers`, but if score files are missing for any real layer, the fetch hangs with "predicts..." text stuck.

### Moderate
6. **Direction depth profile (rv0) Y-scaling** — The late-layer divergence dominates the Y axis, making early-layer micro-divergences invisible. Needs log scale or per-region normalization.

7. **Superposition viewport (rv2) shows only dots at baseline** — The line+dot visualization works but when alignment values are small (most layers), the dots are tiny and hard to see. The `maxSize` calculation uses the max of all dot sizes, which can be very small.

8. **GIF recording flag never resets on browser crash** — If the browser tab crashes during GIF recording, `_recording=true` persists until page reload. The `try/finally` block handles JS errors but not tab crashes.

9. **Score cache eviction during layer sweep** — `_direction_quality` now does an all-layer bimodality sweep using mmap (not cache), but `_direction_depth` and `_direction_circuit` still use `_get_scores_f32` which evicts the 3-layer cache when sweeping all 24 layers. After these calls, the cache is warm for L22-L23 only.

10. **`_updateSeparation` fires on every layer change during playback** — When scrubbing layers with arrow keys, each layer triggers an async server request for direction-quality (which does a full-K matmul + random baseline + all-layer sweep). This floods the server. Should debounce.

11. **Token search results don't show prediction or direction info** — Search finds tokens by text but doesn't show which side of the current direction they're on or what they predict.

12. **MCP GIF timeout hardcoded at 300s in mcp.py** — The MCP server loads code at startup, so changes to timeout values in mcp.py don't take effect until the MCP server restarts (which means restarting Claude Code). The old 90s timeout is still active in running sessions.

### Minor
13. **Overlay text clips on small viewports** — The `_drawOverlay` font sizes are proportional to viewport height, but on the right-column viewports (rv0, rv1, rv2) which are small, the text overlaps.

14. **Direction coloring doesn't reset on model switch** — If you switch models while tokens are pinned, the direction coloring from the old model persists until the new model's full-K projection completes.

15. **`_dirCursor` in rv0 doesn't move when rv0 is rebuilt** — The cursor line for the current layer is added during build but only repositioned in `_updateRVLayer`. If the viewport is rebuilt (new pins), the cursor is recreated but not synced to the current layer until the next layer change.

16. **Flower viewports don't auto-switch to direction alignment** — The `_direction_weight_alignment` endpoint exists but flowers still show PC alignment, not concept-direction alignment, when pins are active.

17. **Chat input in MCP Collaboration panel sends to `/api/navigate`** — The chat feature POSTs `{cmd: 'chat', message: ...}` to the navigate endpoint. The server doesn't handle the `chat` cmd — it just pushes it to the poll queue where nothing processes it.

18. **`_captureLabel` returns wrong label for composite** — Composite captures use the vp0 label in the filename instead of "composite".

## Falsification Results (Tests That Failed)

19. **MLP attribution metric is broken** — MLP=1.0 for every concept at every layer. Random baseline shows MLP is only 3.3σ above random (barely significant). The metric doesn't discriminate concepts from noise. The z-score fix is in but the raw attribution display still shows MLP=1.0 which looks dominant.

20. **Bimodality metric has 4% false positive rate** — Random token pairs produce bimodality <0.3 ("BIMODAL") 4% of the time at L20. The random baseline percentile is now reported but older endpoints (`_direction_depth`, `_auto_discover_directions`) don't include this context.

21. **Layer instability** — Same concept's bimodality varies from 0.14 to 0.79 across layers (Sure/Sorry). The tool now reports the range but older visualizations (rv0 depth profile) don't show bimodality per layer, only magnitude.

22. **Functional hit rate is low for ALL features** — Safety: 50% (partial). Script: 30% (partial). Sentiment: 0%. Gender: 0%. Most directions are geometric projections that don't correlate with functional behavior. The warning is now shown but the "Most X-like" token lists still present as if they're meaningful.

23. **Adversarial bimodality** — Random pair "các"/"メッセージ" achieves bimodality 0.225 ("BIMODAL"). Any Vietnamese/Japanese pair would score similarly because the vocabulary splits by script. Bimodality alone doesn't prove a meaningful feature.

## Features Built But Never Tested

24. **`_direction_weight_alignment` endpoint** — Built, never called from UI, never verified with real data.

25. **`_auto_discover_directions` at template mode** — Only tested at raw mode where crystal dominates. Template mode discovery might find real features but was never run.

26. **`_live_forward` endpoint** — Tested once with Rammstein lyrics. Never integrated into the companion UI. The scores are returned but there's no way to visualize them in the viewer.

27. **Cross-model comparison** — SmolLM2 and Qwen MRIs both exist. `_direction_quality` can run on either. Never compared the same concept between models in a systematic way.

28. **`captureComposite` function** — Written, routing broken (bug #1), never produced a successful composite image.

29. **Circuit attribution for non-safety concepts** — Tested for safety, script, sentiment. Never tested for concrete/abstract, number, agreement, or other concepts.

30. **Steer-test button** — In the UI, never successfully used (requires model loading, which errors if model ID doesn't match HF path).

## Avoided Problems

31. **Never tested what happens when the USB drive disconnects** — The score cache holds float32 arrays from `/Volumes/sharts`. If the drive disconnects, cached data stays valid but new loads will fail with IO errors. No graceful handling.

32. **Never tested with a model larger than 0.5B** — The 3-layer score cache at 537MB per layer = 1.6GB. A 7B model with hidden_dim=4096 would be 150K × 4096 × 4 = 2.4GB per layer. The cache would need adjustment.

33. **Never measured the PCA decomposition's faithfulness** — PCA captures directions that maximize variance. The model might compute in a rotated basis. We project everything into PCA space and assume it's faithful. The reconstruction error of the PCA decomposition is never checked.

34. **Never tested concurrent users** — The companion server is `ThreadingMixIn` but the score cache, poll queue, and capture relay are all shared mutable state. Two browsers connected simultaneously could race.

35. **Never tested template mode predictions (logit lens)** — The `_token_predicts` endpoint loads exit states from the MRI. Template mode exit states might have different shapes or positions than raw mode. Never verified.

36. **The "Discover features" button doesn't exclude the crystal** — In raw mode, every bimodal PC is dominated by 뀔. The tool should exclude extreme outlier tokens before computing bimodality. Never implemented.

37. **Never investigated WHY the crystal exists** — 뀔 dominates every PC in raw mode. Is it the rarest token? A tokenizer bug? A training data artifact? Never looked up its frequency, byte sequence, or training data presence.

## Technical Debt

38. **companion_ui.html is 3700+ lines** — Single file with all JS, HTML, and CSS. Should be split into modules but the single-file design avoids build tooling.

39. **companion.py is 2000+ lines** — Analysis functions (`_direction_quality`, `_direction_circuit`, etc.) should be in a separate module, not in the HTTP server file.

40. **No API versioning** — Endpoints return whatever fields the current code produces. Clients (browser JS, MCP tools) assume specific field names. Adding/removing fields can break the other side silently.

41. **Score cache has no size limit** — `_score_cache` grows unbounded. The LRU eviction removes old layers but lmhead (250MB) stays forever. A model switch doesn't clear the old model's cached scores.

42. **Test coverage gaps** — No tests for: the UI JavaScript, the companion HTTP handler routes, the live-forward endpoint, the capture pipeline, the auto-discover endpoint at template mode, or the circuit z-score computation.

43. **CLAUDE.md is out of date** — Documents Session 6 findings but not Session 10's concept microscope, direction coloring, or any of the new endpoints/tools.

## Next Session Priorities

1. Fix the falsification issues: debounce `_updateSeparation`, integrate random baseline into ALL direction endpoints, make functional validation mandatory
2. Wire `_live_forward` into the companion UI — type a prompt, see tokens appear in the cloud
3. Investigate the crystal (뀔) — frequency, byte sequence, what it tells us about tokenizer design
4. Fix composite capture routing
5. Test at scale: can any of this run on a 3B or 7B model?
6. Update CLAUDE.md with Session 10 documentation
