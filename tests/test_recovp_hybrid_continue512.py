"""CPU regression checks for isolated continuation, recovery and GPU admission."""
import copy
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from paper2.io import read, sha
from tools import recovp_hybrid_continue512 as continuation
from tools import run_recovp_hybrid_continue512 as pipeline
from tools.durable_io import atomic_json


class ToyModel(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 1)

    def cuda(self):
        return self

    def parameter_report(self):
        return {"test_parameters": 3}


class ContinuationTests(unittest.TestCase):
    def test_parent_barrier_requires_matching_completion_receipts(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(continuation, "ROOT", Path(directory)):
            _, evidence, folder = continuation.paths(17)
            self.assertFalse(continuation.parent_ready(17))
            result = {"seed": 17, "epochs": 256,
                      "recovery_sha256": {str(e): "test" for e in range(257)}}
            atomic_json(folder / "TRAINING_COMPLETE.json", result)
            self.assertFalse(continuation.parent_ready(17))
            atomic_json(evidence / "TRAINING_COMPLETE.json", result)
            self.assertTrue(continuation.parent_ready(17))
            atomic_json(evidence / "TRAINING_COMPLETE.json", {**result, "epochs": 512})
            self.assertFalse(continuation.parent_ready(17))

    def test_archive_refuses_a_changed_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = (Path(directory) / name for name in ("source", "archive"))
            source.write_bytes(b"256-epoch model")
            continuation.copy_verified(source, destination)
            continuation.copy_verified(source, destination)
            source.write_bytes(b"different model")
            with self.assertRaisesRegex(RuntimeError, "immutable archive differs"):
                continuation.copy_verified(source, destination)
            self.assertEqual(destination.read_bytes(), b"256-epoch model")

    def test_new_directory_full_state_restore_and_restart(self):
        trainer = continuation.trainer
        model = ToyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        for state in optimizer.state.values():
            state["step"] = torch.tensor(12288.)
        parent = {"format": "parent", "identity": {"original": True}, "seed": 17,
                  "epoch": 256, "global_step": 12288,
                  "model": copy.deepcopy(model.state_dict()),
                  "optimizer": copy.deepcopy(optimizer.state_dict()),
                  "rng": {"torch_cpu": torch.get_rng_state()},
                  "epoch_record": None, "architecture": model.parameter_report()}
        calls = []

        def update(net, optim, arrays, ids, epoch, batch_index, device):
            self.assertEqual(optim.param_groups[0]["lr"], 2e-6)
            optim.zero_grad(set_to_none=True)
            loss = net(torch.ones(1, 2)).square().sum()
            loss.backward()
            optim.step()
            calls.append((epoch, batch_index))
            return {"total": float(loss.detach()), "lr": trainer.learning_rate(epoch, batch_index)}

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch.object(continuation, "ROOT", root))
            stack.enter_context(patch.object(continuation, "FINAL_EPOCH", 258))
            stack.enter_context(patch.object(continuation.admission, "require_server_identity"))
            stack.enter_context(patch.object(continuation.admission, "require_visible_gpu"))
            stack.enter_context(patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
            stack.enter_context(patch.object(torch.cuda, "device_count", return_value=1))
            stack.enter_context(patch.object(torch.cuda, "synchronize"))
            stack.enter_context(patch.object(trainer, "configure_precision", return_value={"test": True}))
            stack.enter_context(patch.object(trainer, "ReCoVPHybrid", ToyModel))
            stack.enter_context(patch.object(trainer, "restore_rng",
                                             side_effect=lambda state: torch.set_rng_state(state["torch_cpu"])))
            stack.enter_context(patch.object(trainer, "rng_snapshot",
                                             side_effect=lambda: {"torch_cpu": torch.get_rng_state()}))
            stack.enter_context(patch.object(trainer, "load_arrays",
                                             return_value={"s": np.empty((512, 8, 1800), dtype=np.uint8)}))
            stack.enter_context(patch.object(trainer, "batches",
                                             side_effect=lambda *_: (np.arange(32) for _ in range(48))))
            stack.enter_context(patch.object(trainer, "optimizer_step", side_effect=update))
            for name in ("FORMAT", "MILESTONES", "learning_rate"):
                stack.enter_context(patch.object(trainer, name, getattr(trainer, name)))
            config_path = continuation.paths(17, True)[0] / "CONFIG.json"

            def freeze(seed, gpu, config, manifest, precision):
                frozen = {"seed": seed, "gpu": gpu, "epochs": 258}
                atomic_json(config_path, frozen)
                return frozen, {"run": "test512", "seed": 17, "config_sha256": sha(config_path)}

            stack.enter_context(patch.object(continuation, "freeze_extension", side_effect=freeze))
            stack.enter_context(patch.object(continuation, "audit_parent", return_value=(
                {}, parent, {"parent_sha256": "original256"})))
            original_folder = continuation.paths(17)[2]
            original_folder.mkdir(parents=True)
            original_checkpoint = original_folder / "epoch_256.pt"
            torch.save(parent, original_checkpoint)
            original_hash = sha(original_checkpoint)
            result = continuation.train(17, 1)
            self.assertEqual(result["epochs"], 258)
            self.assertEqual(calls[0], (257, 0))
            self.assertEqual(calls[-1], (258, 47))
            self.assertEqual(len(calls), 96)
            folder = continuation.paths(17, True)[2]
            endpoint = torch.load(folder / "recovery/epoch_258.pt", weights_only=False)
            for state in endpoint["optimizer"]["state"].values():
                self.assertEqual(state["step"].item(), 12384)
            initial = torch.load(folder / "recovery/epoch_256.pt", weights_only=False)
            for name in parent["model"]:
                self.assertTrue(torch.equal(initial["model"][name], parent["model"][name]))
            self.assertTrue(torch.equal(initial["rng"]["torch_cpu"], parent["rng"]["torch_cpu"]))
            self.assertEqual(sha(original_checkpoint), original_hash)
            # Simulate a crash between checkpoint publication and receipt publication.
            (folder / "recovery/epoch_258.json").unlink()
            continuation.train(17, 1)
            self.assertEqual(len(calls), 96)
            self.assertEqual(sha(original_checkpoint), original_hash)
            self.assertIn("recovered_receipt_utc", read(folder / "recovery/epoch_258.json"))

    def test_controller_waits_for_every_parent(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(pipeline, "STATE", Path(directory)))
            stack.enter_context(patch.object(pipeline, "EXTRA_SOURCES", ()))
            stack.enter_context(patch.object(pipeline.admission, "require_server_identity"))
            stack.enter_context(patch.object(pipeline, "parent_ready", side_effect=lambda seed: seed != 109))
            stack.enter_context(patch.object(pipeline.subprocess, "run"))
            stack.enter_context(patch.object(pipeline, "read", return_value={"parent_sha256": "test"}))
            sleep = stack.enter_context(patch.object(pipeline.time, "sleep", side_effect=InterruptedError))
            start = stack.enter_context(patch.object(pipeline, "start"))
            with self.assertRaises(InterruptedError):
                pipeline.run()
            start.assert_not_called()
            sleep.assert_called_once_with(30)
            self.assertEqual(read(Path(directory) / "STATUS.json")["stage"], "WAITING_FOR_ALL_256")

    def test_controller_dispatches_both_gpus_and_finishes_the_queue(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            state = Path(directory)
            entries = {str(seed): {"stage": "TRAINING_COMPLETE", "attempts": [], "failures": 0}
                       for seed in pipeline.SEEDS}
            for seed in (17, 29):
                entries[str(seed)]["stage"] = "PENDING"
            atomic_json(state / "STATUS.json", {
                "sources": {}, "seeds": entries, "parent_archives_verified_utc": "test"})
            stack.enter_context(patch.object(pipeline, "STATE", state))
            stack.enter_context(patch.object(pipeline.admission, "require_server_identity"))
            stack.enter_context(patch.object(pipeline, "parent_ready", return_value=True))
            stack.enter_context(patch.object(pipeline.time, "sleep"))
            stack.enter_context(patch.object(pipeline, "paths", return_value=(state / "new", state, state)))
            dispatched = []
            stack.enter_context(patch.object(pipeline, "complete", side_effect=lambda seed: len(dispatched) == 2))
            samples = [[{"physical_gpu_index": gpu, "utilization_percent": util}
                        for gpu, util in enumerate(utils)] for utils in ((100, 40), (20, 100))]
            stack.enter_context(patch.object(pipeline.admission, "query_gpus", side_effect=samples))

            def start(seed, gpu, entry):
                entry.update(stage="RUNNING", gpu=gpu, pid=1000 + seed)
                entry["attempts"].append({"pid": 1000 + seed})
                dispatched.append((seed, gpu))
                process = unittest.mock.Mock()
                process.poll.return_value = None
                return process

            stack.enter_context(patch.object(pipeline, "start", side_effect=start))
            result = pipeline.run()
            self.assertEqual(dispatched, [(17, 1), (29, 0)])
            self.assertEqual(result["stage"], "TRAINING_COMPLETE")


if __name__ == "__main__":
    unittest.main()
