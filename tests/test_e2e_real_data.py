"""End-to-end tests with real cached data (skip gracefully if unavailable)."""
import json
import pytest
from pathlib import Path
from heinrich.mcp import ToolServer

DORMANT_INDICES = Path("/Users/asuramaya/Code/carving_machine_v3/conker-detect/out/model-indices/dormant-1")
OPC_SUBMISSION = Path(
    "/Users/asuramaya/Code/carving_machine_v3/opc-parameter-golf-submission"
    "/for_parameter_golf/records/track_non_record_16mb"
    "/2026-03-30_OPC_CausalPackedMemory_NativeFullSpecClean"
)
DELTA_BUNDLE = Path(
    "/Users/asuramaya/Code/carving_machine_v3/conker-detect/out/delta-bundles/d1_qa_only.npz"
)


@pytest.mark.skipif(not DORMANT_INDICES.exists(), reason="dormant puzzle data not available")
def test_e2e_fetch_dormant_model():
    server = ToolServer()
    result = server.call_tool("heinrich_fetch", {"source": str(DORMANT_INDICES), "label": "dormant-1"})
    assert result["signals_summary"]["total"] > 0
    assert result["structural"]["architecture_type"] == "deepseek_v3"


@pytest.mark.skipif(not DELTA_BUNDLE.exists(), reason="delta bundle not available")
def test_e2e_inspect_delta():
    server = ToolServer()
    result = server.call_tool("heinrich_inspect", {"source": str(DELTA_BUNDLE), "label": "d1-delta"})
    assert result["signals_summary"]["total"] > 0
    spectral = server.call_tool("heinrich_signals", {"kind": "spectral_sigma1"})
    assert spectral["count"] > 0


@pytest.mark.skipif(not OPC_SUBMISSION.exists(), reason="OPC submission not available")
def test_e2e_validate_opc():
    server = ToolServer()
    result = server.call_tool("heinrich_validate", {"source": str(OPC_SUBMISSION), "label": "opc"})
    assert result["signals_summary"]["total"] > 0
    consistency = server.call_tool("heinrich_signals", {"kind": "cross_file_consistency"})
    assert consistency["count"] > 0


@pytest.mark.skipif(not OPC_SUBMISSION.exists(), reason="OPC submission not available")
def test_e2e_compete_opc():
    server = ToolServer()
    result = server.call_tool("heinrich_compete", {
        "source": str(OPC_SUBMISSION),
        "profile": "parameter-golf",
        "label": "opc",
    })
    assert result["signals_summary"]["total"] > 0
    rules = server.call_tool("heinrich_signals", {"kind": "rule_check"})
    assert rules["count"] > 0


def test_e2e_observe_loop_mock():
    """Always runs — uses mock data."""
    server = ToolServer()
    server.call_tool("heinrich_observe", {"grid": [[0, 1, 2], [1, 0, 1], [2, 1, 0]], "label": "test"})
    server.call_tool("heinrich_observe", {"grid": [[1, 1, 1], [1, 1, 1], [1, 1, 1]], "label": "test"})
    bundle = server.call_tool("heinrich_bundle", {})
    assert bundle["signals_summary"]["total"] > 0
    assert len(bundle.get("findings", [])) >= 0
