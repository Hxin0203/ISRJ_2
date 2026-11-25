"""Exercise the post-training controller without CUDA, SSH, or model work."""
from types import SimpleNamespace

import pytest

from paper2.io import read, sha
from tools import finish_recovp_hybrid_seed17 as finish
from tools import server_gpu_admission as admission
from tools.durable_io import ProcessLock, atomic_json


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    evidence = tmp_path / "evidence" / "outputs" / finish.RUN
    folder = tmp_path / "outputs" / finish.RUN / "seed_17"
    evidence.mkdir(parents=True)
    folder.mkdir(parents=True)
    for name, value in {
        "ROOT": tmp_path,
        "EVIDENCE": evidence,
        "FOLDER": folder,
        "COMPLETE": folder / "TRAINING_COMPLETE.json",
        "RESULT": folder / "dev_epoch_256_v1.json",
        "STATUS": evidence / "POSTTRAINING_STATUS.json",
    }.items():
        monkeypatch.setattr(finish, name, value)
    monkeypatch.setattr(finish, "require_server_identity", lambda: None)
    monkeypatch.setattr(finish, "source_bindings", lambda: {"controller.py": "frozen-source"})
    monkeypatch.setattr(finish, "process_alive", lambda pid: True)
    (folder / "PROCESS.lock").write_bytes(b'\0{"pid": 12345, "acquired_unix": 1234}')
    atomic_json(evidence / "STATUS.json", {"stage": "TRAINING", "epoch": 255})
    return tmp_path


def complete_training(**overrides):
    receipt = {
        "stage": "TRAINING_COMPLETE", "seed": 17, "epochs": 256,
        "identity": {"run": finish.RUN, "seed": 17, "source_sha256": {"model.py": "model-sha"}},
        "milestones": {"256": "checkpoint-sha"},
        "recovery_sha256": {"256": "checkpoint-sha"},
        **overrides,
    }
    atomic_json(finish.COMPLETE, receipt)
    return receipt


def publish_evaluation(**overrides):
    complete = read(finish.COMPLETE)
    raw = finish.FOLDER / "dev_epoch_256_parent_values.npz"
    raw.write_bytes(b"offline fixture for archive integrity")
    result = {
        "stage": "DEV_ONLY_RECOVP_HYBRID", "seed": 17, "epoch": 256, "parents": 256,
        "checkpoint_identity": complete["identity"],
        "checkpoint_sha256": complete["milestones"]["256"],
        "raw_parent_values_sha256": sha(raw),
        **overrides,
    }
    atomic_json(finish.RESULT, result)
    return result


def gpu_inventory(util0, util1):
    return [{"physical_gpu_index": index, "gpu_uuid": admission.GPU_UUIDS[index],
             "gpu_name": f"GPU {index}", "utilization_percent": util}
            for index, util in enumerate((util0, util1))]


def successful_evaluator(calls):
    def evaluate(command, **kwargs):
        calls.append((command, kwargs))
        assert command == [finish.sys.executable, "-m", "tools.vp_1103_recovp_hybrid_evaluate",
                           "--batch-parents", "2"]
        assert kwargs["cwd"] == finish.ROOT
        publish_evaluation()
        return SimpleNamespace(returncode=0)
    return evaluate


def test_waits_for_completion_receipt_even_if_epoch_256_is_saved(workspace, monkeypatch):
    atomic_json(finish.EVIDENCE / "STATUS.json", {"stage": "EPOCH_SAVED", "epoch": 256})
    waits, calls = [], []

    def sleep(seconds):
        waits.append(read(finish.STATUS))
        assert seconds == 5
        assert not finish.COMPLETE.exists()
        complete_training()

    def choose():
        assert finish.COMPLETE.exists(), "GPU admission must wait for TRAINING_COMPLETE"
        return admission.select_gpu(gpu_inventory(20, 100))

    monkeypatch.setattr(finish.time, "sleep", sleep)
    monkeypatch.setattr(finish, "choose_gpu", choose)
    monkeypatch.setattr(finish.subprocess, "run", successful_evaluator(calls))
    result = finish.run(poll_seconds=5)
    assert [row["stage"] for row in waits] == ["WAITING_FOR_SEED17_TRAINING"]
    assert len(calls) == 1
    assert result["stage"] == "CORE_DEV_COMPLETE_REVIEW_REQUIRED"
    assert result["remaining_task_book_work"]
    assert any("equal-PFA" in item for item in result["remaining_task_book_work"])
    assert any("SUMMARY/REPORT/VERIFICATION/REPRODUCE" in item
               for item in result["remaining_task_book_work"])


@pytest.mark.parametrize("util0,util1,expected_gpu", [(20, 100, "0"), (100, 20, "1"),
                                                     (84.999, 85, "0"), (85, 84.999, "1")])
def test_dispatch_uses_either_admitted_gpu_without_launching_training(
        workspace, monkeypatch, util0, util1, expected_gpu):
    complete_training()
    calls = []
    monkeypatch.setattr(finish, "choose_gpu", lambda: admission.select_gpu(gpu_inventory(util0, util1)))
    monkeypatch.setattr(finish.subprocess, "run", successful_evaluator(calls))
    finish.run(poll_seconds=5)
    assert len(calls) == 1
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == expected_gpu


def test_waits_when_both_gpus_busy_then_evaluates_once(workspace, monkeypatch):
    complete_training()
    calls, waits = [], []
    inventories = iter([gpu_inventory(85, 100), gpu_inventory(40, 99)])
    monkeypatch.setattr(finish, "choose_gpu", lambda: admission.select_gpu(next(inventories)))
    monkeypatch.setattr(finish.time, "sleep", lambda seconds: waits.append(read(finish.STATUS)["stage"]))
    monkeypatch.setattr(finish.subprocess, "run", successful_evaluator(calls))
    finish.run(poll_seconds=5)
    assert waits == ["WAITING_FOR_GPU_BELOW_85_PERCENT"]
    assert len(calls) == 1


@pytest.mark.parametrize("overrides", [{"stage": "EPOCH_SAVED"}, {"seed": 29}, {"epochs": 255}])
def test_invalid_completion_receipt_fails_before_dispatch(workspace, monkeypatch, overrides):
    complete_training(**overrides)
    monkeypatch.setattr(finish, "choose_gpu", lambda: pytest.fail("must not admit a GPU"))
    monkeypatch.setattr(finish.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not launch"))
    with pytest.raises(RuntimeError, match="invalid seed-17 training completion"):
        finish.run(poll_seconds=5)


def test_existing_valid_evaluation_skips_dispatch_and_reports_remaining_work(workspace, monkeypatch):
    complete_training()
    publish_evaluation()
    monkeypatch.setattr(finish, "choose_gpu", lambda: pytest.fail("finished evaluation needs no GPU"))
    monkeypatch.setattr(finish.subprocess, "run", lambda *args, **kwargs: pytest.fail("duplicate evaluation"))
    result = finish.run(poll_seconds=5)
    assert result["stage"] == "CORE_DEV_COMPLETE_REVIEW_REQUIRED"
    assert result["remaining_task_book_work"]


@pytest.mark.parametrize("corruption", ["identity", "raw_archive"])
def test_existing_corrupt_evaluation_is_never_overwritten(workspace, monkeypatch, corruption):
    complete_training()
    if corruption == "identity":
        publish_evaluation(checkpoint_sha256="different-checkpoint")
    else:
        publish_evaluation()
        (finish.FOLDER / "dev_epoch_256_parent_values.npz").write_bytes(b"changed")
    monkeypatch.setattr(finish.subprocess, "run", lambda *args, **kwargs: pytest.fail("must preserve invalid evidence"))
    with pytest.raises(RuntimeError, match="identity differs|archive SHA is invalid"):
        finish.run(poll_seconds=5)


@pytest.mark.parametrize("error", ["RuntimeError: nonfinite model output", "RuntimeError: source binding stale",
                                    "torch.OutOfMemoryError: CUDA out of memory"])
def test_numerical_identity_and_memory_errors_fail_without_retry(workspace, monkeypatch, error):
    complete_training()
    calls = []
    monkeypatch.setattr(finish, "choose_gpu", lambda: admission.select_gpu(gpu_inventory(10, 100)))
    monkeypatch.setattr(finish.time, "sleep", lambda seconds: pytest.fail("must not retry model/identity failure"))

    def fail(command, **kwargs):
        calls.append(command)
        kwargs["stderr"].write(error.encode())
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(finish.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="fixed DEV evaluation exited"):
        finish.run(poll_seconds=5)
    assert len(calls) == 1
    status = read(finish.STATUS)
    assert status["stage"] == "DEV_EVALUATION_FAILED"
    assert status["returncode"] == 1
    assert error in status["error_tail"]
    assert "fixed DEV evaluation exited" in status["error"]["message"]


def test_admission_race_can_retry_before_any_evaluation_output(workspace, monkeypatch):
    complete_training()
    calls, waits = [], []
    monkeypatch.setattr(finish, "choose_gpu", lambda: admission.select_gpu(gpu_inventory(20, 100)))
    monkeypatch.setattr(finish.time, "sleep", lambda seconds: waits.append(read(finish.STATUS)["stage"]))

    def evaluate(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            kwargs["stderr"].write(b"tools.server_gpu_admission.NoGpuAvailable: selected GPU at or above 85 percent\n")
            return SimpleNamespace(returncode=1)
        publish_evaluation()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(finish.subprocess, "run", evaluate)
    finish.run(poll_seconds=5)
    assert waits == ["WAITING_FOR_GPU_BELOW_85_PERCENT"]
    assert len(calls) == 2


def test_stale_admission_error_does_not_make_current_numerical_failure_retry(workspace, monkeypatch):
    complete_training()
    (finish.EVIDENCE / "dev.stderr.log").write_bytes(
        b"tools.server_gpu_admission.NoGpuAvailable: old admission failure\n")
    calls = []
    monkeypatch.setattr(finish, "choose_gpu", lambda: admission.select_gpu(gpu_inventory(20, 100)))
    monkeypatch.setattr(finish.time, "sleep", lambda seconds: pytest.fail("stale admission log caused a retry"))

    def fail(command, **kwargs):
        calls.append(command)
        kwargs["stderr"].write(b"RuntimeError: nonfinite model output\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(finish.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="fixed DEV evaluation exited"):
        finish.run(poll_seconds=5)
    assert len(calls) == 1


def test_success_without_result_is_not_relaunched(workspace, monkeypatch):
    complete_training()
    calls = []
    monkeypatch.setattr(finish, "choose_gpu", lambda: admission.select_gpu(gpu_inventory(20, 100)))

    def missing_result(command, **kwargs):
        calls.append(command)
        assert len(calls) == 1, "successful evaluator without an artifact was relaunched"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(finish.subprocess, "run", missing_result)
    with pytest.raises(RuntimeError, match="result|artifact|publish|output"):
        finish.run(poll_seconds=5)
    assert len(calls) == 1


def test_second_controller_cannot_launch_while_process_lock_is_held(workspace, monkeypatch):
    complete_training()
    atomic_json(finish.STATUS, {"stage": "RUNNING_FIXED_DEV256", "controller_pid": 111})
    monkeypatch.setattr(finish.subprocess, "run", lambda *args, **kwargs: pytest.fail("duplicate controller launched evaluation"))
    with ProcessLock(finish.EVIDENCE / "POSTTRAINING_PROCESS.lock"):
        with pytest.raises(RuntimeError, match="another process holds"):
            finish.run(poll_seconds=5)
    assert read(finish.STATUS) == {"stage": "RUNNING_FIXED_DEV256", "controller_pid": 111}


def test_source_change_during_wait_fails_before_dispatch(workspace, monkeypatch):
    versions = iter([{"controller.py": "original"}, {"controller.py": "changed"}])
    monkeypatch.setattr(finish, "source_bindings", lambda: next(versions))
    monkeypatch.setattr(finish.subprocess, "run", lambda *args, **kwargs: pytest.fail("changed source was dispatched"))
    with pytest.raises(RuntimeError, match="source changed during wait"):
        finish.run(poll_seconds=5)
    status = read(finish.STATUS)
    assert status["stage"] == "POSTTRAINING_FAILED"
    assert status["error"]["type"] == "RuntimeError"
    assert "source changed during wait" in status["error"]["traceback"]


def test_dead_training_pid_fails_durably_without_restart(workspace, monkeypatch):
    monkeypatch.setattr(finish, "process_alive", lambda pid: False)
    monkeypatch.setattr(finish.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not restart training"))
    monkeypatch.setattr(finish.time, "sleep", lambda seconds: pytest.fail("dead trainer must not wait forever"))
    with pytest.raises(RuntimeError, match="stopped without TRAINING_COMPLETE"):
        finish.run(poll_seconds=5)
    status = read(finish.STATUS)
    assert status["stage"] == "POSTTRAINING_FAILED"
    assert "stopped without TRAINING_COMPLETE" in status["error"]["message"]


@pytest.mark.parametrize("lock_bytes", [None, b"\0not-json", b'\0{"pid": 0}'])
def test_unreadable_training_lock_retries_three_times_then_fails_durably(
        workspace, monkeypatch, lock_bytes):
    path = finish.FOLDER / "PROCESS.lock"
    if lock_bytes is None:
        path.unlink()
    else:
        path.write_bytes(lock_bytes)
    waits = []
    monkeypatch.setattr(finish.time, "sleep", waits.append)
    monkeypatch.setattr(finish.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not dispatch"))
    with pytest.raises(RuntimeError, match="cannot read valid seed-17 training lock metadata"):
        finish.run(poll_seconds=5)
    assert waits == [0.1, 0.1]
    assert read(finish.STATUS)["stage"] == "POSTTRAINING_FAILED"


def test_completion_published_during_dead_pid_check_avoids_false_failure(workspace, monkeypatch):
    calls = []

    def trainer_finished(pid):
        complete_training()
        return False

    monkeypatch.setattr(finish, "process_alive", trainer_finished)
    monkeypatch.setattr(finish, "choose_gpu", lambda: admission.select_gpu(gpu_inventory(20, 100)))
    monkeypatch.setattr(finish.subprocess, "run", successful_evaluator(calls))
    result = finish.run(poll_seconds=5)
    assert result["stage"] == "CORE_DEV_COMPLETE_REVIEW_REQUIRED"
    assert len(calls) == 1


def test_initial_source_hash_error_is_durable(workspace, monkeypatch):
    def missing_source():
        raise FileNotFoundError("controller source is missing")

    monkeypatch.setattr(finish, "source_bindings", missing_source)
    with pytest.raises(FileNotFoundError, match="source is missing"):
        finish.run(poll_seconds=5)
    status = read(finish.STATUS)
    assert status["stage"] == "POSTTRAINING_FAILED"
    assert status["error"]["type"] == "FileNotFoundError"


def test_subprocess_launch_exception_is_durable(workspace, monkeypatch):
    complete_training()
    monkeypatch.setattr(finish, "choose_gpu", lambda: admission.select_gpu(gpu_inventory(20, 100)))

    def launch_failed(*args, **kwargs):
        raise OSError("cannot create evaluator process")

    monkeypatch.setattr(finish.subprocess, "run", launch_failed)
    with pytest.raises(OSError, match="cannot create evaluator"):
        finish.run(poll_seconds=5)
    status = read(finish.STATUS)
    assert status["stage"] == "POSTTRAINING_FAILED"
    assert status["previous_status"]["stage"] == "RUNNING_FIXED_DEV256"
    assert status["error"]["type"] == "OSError"


def test_liveness_probe_reads_after_lock_byte_and_preserves_metadata(workspace, monkeypatch):
    path = finish.FOLDER / "PROCESS.lock"
    original = path.read_bytes()
    seen = []
    monkeypatch.setattr(finish, "process_alive", lambda pid: seen.append(pid) or True)
    assert finish.training_liveness()["pid_alive"] is True
    assert seen == [12345]
    assert path.read_bytes() == original
