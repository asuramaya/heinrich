"""Tests for heinrich.cartography.datasets — dataset manager with caching.

Phase 4 (Principle 8): No silent fallback to built-in prompts. All loads
must come from cache or HuggingFace, raising on failure.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from heinrich.cartography.datasets import (
    load_dataset,
    load_all,
    list_datasets,
    register_dataset,
    _REGISTRY,
    _load_cache,
    _save_cache,
    CACHE_DIR,
)


class TestRegistry:
    def test_default_datasets_present(self):
        names = list_datasets()
        assert "simple_safety" in names
        assert "harmbench" in names
        assert "do_not_answer" in names
        assert "xstest" in names
        assert "catqa" in names
        assert "forbidden_questions" in names
        # sorry_bench and wildguard removed: gated datasets requiring auth
        # Phase 4 additions
        assert "simplesafetytests" in names
        assert "toxicchat" in names

    def test_list_datasets_sorted(self):
        names = list_datasets()
        assert names == sorted(names)

    def test_register_new_dataset(self):
        register_dataset("test_ds", "org/test-ds", "train", "text", "label")
        assert "test_ds" in list_datasets()
        spec = _REGISTRY["test_ds"]
        assert spec.hf_id == "org/test-ds"
        assert spec.split == "train"
        assert spec.prompt_column == "text"
        assert spec.category_column == "label"
        # cleanup
        del _REGISTRY["test_ds"]

    def test_register_override(self):
        old_hf_id = _REGISTRY["simple_safety"].hf_id
        register_dataset("simple_safety", "new/id", "test", "p", "c")
        assert _REGISTRY["simple_safety"].hf_id == "new/id"
        # restore
        register_dataset("simple_safety", old_hf_id, "test", "prompt", "harm_area")


class TestCaching:
    def test_save_and_load_cache(self, tmp_path):
        prompts = [
            {"prompt": "test1", "category": "cat1", "source": "ds1"},
            {"prompt": "test2", "category": "cat2", "source": "ds1"},
        ]
        with patch("heinrich.cartography.datasets.CACHE_DIR", tmp_path):
            _save_cache("test_ds", prompts)
            cached = _load_cache("test_ds")
        assert cached is not None
        assert len(cached) == 2
        assert cached[0]["prompt"] == "test1"

    def test_load_cache_missing(self, tmp_path):
        with patch("heinrich.cartography.datasets.CACHE_DIR", tmp_path):
            result = _load_cache("nonexistent")
        assert result is None

    def test_load_cache_corrupt_json(self, tmp_path):
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("not valid json {{{")
        with patch("heinrich.cartography.datasets.CACHE_DIR", tmp_path):
            result = _load_cache("corrupt")
        assert result is None

    def test_load_cache_empty_list(self, tmp_path):
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("[]")
        with patch("heinrich.cartography.datasets.CACHE_DIR", tmp_path):
            result = _load_cache("empty")
        assert result is None

    def test_cache_file_format(self, tmp_path):
        prompts = [{"prompt": "hello", "category": "test", "source": "x"}]
        with patch("heinrich.cartography.datasets.CACHE_DIR", tmp_path):
            _save_cache("my_ds", prompts)
        path = tmp_path / "my_ds.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == prompts


class TestLoadDataset:
    def test_unknown_dataset_raises(self):
        """Phase 4: unknown dataset must raise RuntimeError, not fallback."""
        with pytest.raises(RuntimeError, match="Unknown dataset"):
            load_dataset("totally_unknown_dataset_xyz")

    def test_cache_hit_skips_hf(self, tmp_path):
        """When cache exists, should not attempt HF download."""
        cached = [
            {"prompt": "cached1", "category": "c1", "source": "simple_safety"},
            {"prompt": "cached2", "category": "c2", "source": "simple_safety"},
        ]
        with patch("heinrich.cartography.datasets.CACHE_DIR", tmp_path):
            _save_cache("simple_safety", cached)
            with patch("heinrich.cartography.datasets._fetch_hf") as mock_fetch:
                result = load_dataset("simple_safety")
        mock_fetch.assert_not_called()
        assert len(result) == 2
        assert result[0]["prompt"] == "cached1"

    def test_cache_miss_triggers_hf(self, tmp_path):
        """When no cache, should call _fetch_hf."""
        hf_result = [{"prompt": "from_hf", "category": "hf_cat", "source": "simple_safety"}]
        with patch("heinrich.cartography.datasets.CACHE_DIR", tmp_path):
            with patch("heinrich.cartography.datasets._fetch_hf", return_value=hf_result) as mock_fetch:
                result = load_dataset("simple_safety")
        mock_fetch.assert_called_once()
        assert result[0]["prompt"] == "from_hf"
        # Should also cache the result
        cache_file = tmp_path / "simple_safety.json"
        assert cache_file.exists()

    def test_hf_failure_raises(self, tmp_path):
        """Phase 4: when HF fails, should raise RuntimeError, not fallback."""
        with patch("heinrich.cartography.datasets.CACHE_DIR", tmp_path):
            with patch("heinrich.cartography.datasets._fetch_hf", side_effect=RuntimeError("download failed")):
                with pytest.raises(RuntimeError, match="download failed"):
                    load_dataset("simple_safety")

    def test_no_cache_mode(self, tmp_path):
        """When cache=False, should skip cache check and always call HF."""
        cached = [{"prompt": "cached", "category": "c", "source": "simple_safety"}]
        hf_result = [{"prompt": "fresh", "category": "c", "source": "simple_safety"}]
        with patch("heinrich.cartography.datasets.CACHE_DIR", tmp_path):
            _save_cache("simple_safety", cached)
            with patch("heinrich.cartography.datasets._fetch_hf", return_value=hf_result):
                result = load_dataset("simple_safety", cache=False)
        assert result[0]["prompt"] == "fresh"

    def test_max_prompts_with_cache(self, tmp_path):
        """max_prompts should limit results from cache."""
        cached = [
            {"prompt": f"p{i}", "category": "c", "source": "simple_safety"}
            for i in range(10)
        ]
        with patch("heinrich.cartography.datasets.CACHE_DIR", tmp_path):
            _save_cache("simple_safety", cached)
            result = load_dataset("simple_safety", max_prompts=3)
        assert len(result) == 3


class TestLoadAll:
    def test_load_all_aggregates(self, tmp_path):
        """load_all should return prompts from all datasets."""
        fake_prompts = [{"prompt": "p1", "category": "c", "source": "ds"}]
        with patch("heinrich.cartography.datasets.load_dataset", return_value=fake_prompts) as mock_ld:
            result = load_all(max_per_dataset=10)
        n_datasets = len(list_datasets())
        assert mock_ld.call_count == n_datasets
        assert len(result) == n_datasets  # 1 prompt per dataset

    def test_load_all_passes_max(self):
        """max_per_dataset should be forwarded."""
        with patch("heinrich.cartography.datasets.load_dataset", return_value=[]) as mock_ld:
            load_all(max_per_dataset=25)
        for call in mock_ld.call_args_list:
            assert call.kwargs["max_prompts"] == 25


class TestNoBuiltinFallback:
    """Phase 4: verify no hardcoded prompts or silent fallback exist."""

    def test_no_builtin_prompts_constant(self):
        import heinrich.cartography.datasets as mod
        assert not hasattr(mod, "_BUILTIN_PROMPTS")

    def test_no_fallback_function(self):
        import heinrich.cartography.datasets as mod
        assert not hasattr(mod, "_fallback")

    def test_import_error_when_no_datasets_lib(self, tmp_path):
        """Phase 4: raise ImportError with install instructions when datasets not installed."""
        with patch("heinrich.cartography.datasets.CACHE_DIR", tmp_path):
            with patch.dict("sys.modules", {"datasets": None}):
                with patch("builtins.__import__", side_effect=ImportError("No module named 'datasets'")):
                    with pytest.raises((ImportError, RuntimeError)):
                        load_dataset("simple_safety")

    def test_runtime_error_on_network_failure(self, tmp_path):
        """Phase 4: raise RuntimeError with clear message on network failure."""
        with patch("heinrich.cartography.datasets.CACHE_DIR", tmp_path):
            with patch(
                "heinrich.cartography.datasets._fetch_hf",
                side_effect=RuntimeError("Failed to download dataset 'Bertievidgen/SimpleSafetyTests': ConnectionError"),
            ):
                with pytest.raises(RuntimeError, match="Failed to download"):
                    load_dataset("simple_safety")
