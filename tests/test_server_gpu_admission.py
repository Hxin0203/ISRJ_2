"""Scheduling boundaries and identity checks without server or CUDA access."""
import importlib
import os
from unittest.mock import patch

import pytest

from tools import server_gpu_admission as admission


def inventory(util0, util1):
    return [{"physical_gpu_index": index, "gpu_uuid": admission.GPU_UUIDS[index],
             "gpu_name": f"GPU {index}", "utilization_percent": utilization}
            for index, utilization in enumerate((util0, util1))]


def test_85_percent_is_not_admitted_and_alternate_is_selected():
    selected = admission.select_gpu(inventory(85, 84.999))
    assert selected["physical_gpu_index"] == 1


def test_busy_gpu1_does_not_block_gpu0():
    assert admission.select_gpu(inventory(20, 100))["physical_gpu_index"] == 0


def test_least_utilized_is_selected_without_idle_requirement():
    assert admission.select_gpu(inventory(70, 65))["physical_gpu_index"] == 1


def test_both_busy_requires_waiting():
    with pytest.raises(admission.NoGpuAvailable):
        admission.select_gpu(inventory(85, 100))


def test_uuid_mismatch_is_rejected_even_if_idle():
    samples = inventory(0, 0)
    samples[0]["gpu_uuid"] = "GPU-other"
    with pytest.raises(RuntimeError, match="UUID mismatch"):
        admission.select_gpu(samples)


def test_missing_physical_device_is_rejected():
    with pytest.raises(RuntimeError, match="both physical GPUs"):
        admission.select_gpu(inventory(0, 0)[:1])


def test_inventory_parses_nvidia_smi_csv():
    output = "\n".join(f"{r['physical_gpu_index']}, {r['gpu_uuid']}, RTX 5060 Ti, {r['utilization_percent']}"
                       for r in inventory(73, 84))
    assert admission.parse_gpu_inventory(output) == [
        {**r, "gpu_name": "RTX 5060 Ti"} for r in inventory(73, 84)]


def test_visible_gpu_rechecks_requested_device_instead_of_switching_silently(monkeypatch):
    monkeypatch.setattr(admission, "query_gpus", lambda: inventory(0, 85))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(admission.NoGpuAvailable):
        admission.require_visible_gpu()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert admission.require_visible_gpu()["physical_gpu_index"] == 0


def test_query_is_blocked_on_wrong_host_before_running_a_command(monkeypatch):
    monkeypatch.setattr(admission.platform, "node", lambda: "local-workstation")
    with patch.object(admission.subprocess, "check_output", side_effect=AssertionError("must not run")):
        with pytest.raises(RuntimeError, match="restricted to server B"):
            admission.query_gpus()


def test_query_is_blocked_on_wrong_root_before_running_a_command(monkeypatch, tmp_path):
    monkeypatch.setattr(admission.platform, "node", lambda: admission.SERVER_HOST)
    monkeypatch.setattr(admission, "ROOT", tmp_path)
    with patch.object(admission.subprocess, "check_output", side_effect=AssertionError("must not run")):
        with pytest.raises(RuntimeError, match="root must be"):
            admission.query_gpus()


def test_import_performs_no_commands_or_environment_changes():
    before = os.environ.copy()
    with patch.object(admission.subprocess, "check_output", side_effect=AssertionError("must not run")):
        importlib.reload(admission)
    assert os.environ == before
