def test_inspect_exports():
    from heinrich.inspect import analyze_matrix, analyze_logits, scan_file, load_lora_deltas, load_tensors
    assert callable(analyze_matrix)
    assert callable(analyze_logits)
    assert callable(scan_file)
    assert callable(load_lora_deltas)
    assert callable(load_tensors)

def test_diff_exports():
    from heinrich.diff import patch_weights, merge_weights, decompose_heads
    assert callable(patch_weights)
    assert callable(merge_weights)
    assert callable(decompose_heads)

def test_probe_exports():
    from heinrich.probe import compute_steering_vector, MockEnvironment, SelfAnalyzeStage, HuggingFaceLocalProvider
    assert callable(compute_steering_vector)

def test_bundle_exports():
    from heinrich.bundle import get_profile, apply_profile, package_zip, generate_triage_report, scan_directory, infer_claim_level
    assert callable(get_profile)
    assert callable(generate_triage_report)

def test_top_level_exports():
    from heinrich import Signal, SignalStore, Pipeline, Stage, ToolServer
    assert callable(Signal)
