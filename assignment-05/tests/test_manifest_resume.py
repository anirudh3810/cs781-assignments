"""Regression tests for finalization after a previous Q4-Q6 handoff.

Fixtures are temporary. Reported numerical values below test validation logic;
they are not assignment results or trained-model claims.
"""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
import torch
import a5_runtime as R
import a5_experiments as E
import common_a5 as C
import grokking as G
import model_arch as MA


class ManifestResumeTests(unittest.TestCase):
    def setUp(self):
        self.previous_cwd = Path.cwd()
        self.temp = tempfile.TemporaryDirectory(dir=WORKSPACE)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(os.chdir, self.previous_cwd)
        os.chdir(self.temp.name)
        for name in ("model_arch.py", "attention.py", "rope.py", "grokking.py"):
            shutil.copy2(WORKSPACE / name, name)
        self.env = patch.dict(os.environ, {"A5_ARTIFACT_ROOT": str(Path.cwd() / "artifacts")})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.root = R.artifact_root()
        self.root_patch = patch.object(R, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.cfg = MA.LMConfig(vocab_size=8, d_model=8, n_layers=1, n_heads=2,
                               n_kv_heads=2, d_ff=16, max_seq_len=8, dropout=0.)
        self.model = MA.DecoderLM(self.cfg)
        with torch.no_grad():
            self.model.tok_emb.weight[0, 0] = 1.
        self.memo = copy.deepcopy(self.model)
        with torch.no_grad():
            self.memo.tok_emb.weight[0, 0] = 0.
        self.train = torch.ones((2, 9), dtype=torch.long)
        self.eval = torch.ones((3, 9), dtype=torch.long)
        self.history = {"step": [0, 100, 200], "train_acc_repeat": [.1, .95, .99],
                        "test_acc_repeat": [.1, .4, .96]}
        for name, model in (("model.pt", self.model), ("memorised.pt", self.memo)):
            C.save_decoder_model("submissions/q1", model, name,
                                 architecture_source=["model_arch.py", "attention.py", "rope.py"])
        Path("submissions/q1/tokenizer.json").write_text("{}", encoding="utf-8")
        C.save_split("submissions/q1", train=self.train, eval=self.eval, P=4)
        C.save_metrics("submissions/q1", {"history": self.history, "grokking_gap": G.grokking_gap(self.history)})

    @staticmethod
    def evaluate(model, tokens, passage_length):
        marker = round(float(model.tok_emb.weight[0, 0].detach()))
        return {"acc_repeat": (.99 if marker == 1 else .86) if len(tokens) == 2
                else {0: .20, 1: .96, 2: .80}[marker], "acc_first": .20}

    def test_finalize_twice_with_mixed_manifest_schemas(self):
        R.mark_complete("q1", "q1_repeat_baseline", "fixture", {}, {})
        directory = self.root / "manifests"
        handoff = {"source": "submissions/q1", "target": "submissions/q4_q6_model",
                   "files_sha256": {}, "heldout_repeat_accuracy": .96}
        R.atomic_json_save(handoff, directory / "q1_to_q4_q6_handoff.json")
        R.atomic_json_save({"experiment_id": "handoff", "files_sha256": {}}, directory / "q1_legacy_handoff.json")
        R.atomic_json_save({"experiment_id": "broken_stage", "fingerprint": "fixture", "state": "missing.pt"},
                          directory / "q1_incomplete_stage.json")
        (directory / "q1_corrupt.json").write_text("{invalid JSON", encoding="utf-8")
        with patch.object(G, "evaluate_repeat", side_effect=self.evaluate):
            first = E.finalize_question("q1")
            second = E.finalize_question("q1")
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["experiments"]["q1_repeat_baseline"], "complete")
        self.assertEqual(first["experiments"]["broken_stage"], "invalid_manifest")
        self.assertIn("q1_to_q4_q6_handoff.json", first["auxiliary_manifests"])
        self.assertIn("q1_legacy_handoff.json", first["auxiliary_manifests"])
        self.assertEqual(len(first["manifest_issues"]), 2)

    def test_recovery_reuses_completed_checkpoint_without_training(self):
        late = copy.deepcopy(self.model.state_dict())
        late["tok_emb.weight"][0, 0] = 2.
        late["head.weight"][0, 0] = 2.
        late["__step__"] = 100
        early = copy.deepcopy(self.memo.state_dict())
        early["__step__"] = 50
        exp_id = "q1_pair_recovery_seed_0"
        config = E._q1_repeat_config(exp_id, self.train, self.eval, self.model, 4, 5000, 1e-3, 1., 3.)
        config.update(seed=0, observation_interval=10, memorisation_threshold=.85)
        fp = E._config_fp(config)
        checkpoint = R.checkpoint_path("q1", exp_id)
        R.atomic_torch_save({"fingerprint": fp}, checkpoint)
        R.mark_complete("q1", exp_id, fp, {"model": self.model.state_dict(),
                        "history": self.history, "mem_state": early}, config, required=[checkpoint])
        with patch.dict(os.environ, {"A5_Q1_REPAIR_SEEDS": "0"}), \
                patch.object(G, "evaluate_repeat", side_effect=self.evaluate), \
                patch.object(G, "train_grokking", side_effect=AssertionError("Retraining must not occur")):
            _, _, _, selected = E.ensure_q1_submission_pair(
                G, self.model, self.history, late, self.train, self.eval, 4)
        self.assertEqual(selected, exp_id)
        validation = json.loads((self.root / "metrics/q1/q1_pair_validation.json").read_text())
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["selected_experiment"], exp_id)

    def test_missing_official_artifacts_are_pending(self):
        Path("submissions/q1/memorised.pt").unlink()
        for question in ("q1", "q2", "q3"):
            R.atomic_json_save({"experiment_id": "auxiliary"},
                              self.root / "manifests" / f"{question}_auxiliary.json")
            result = E.finalize_question(question)
            self.assertEqual(result["status"], "pending")
            self.assertTrue(result["missing_official_files"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
