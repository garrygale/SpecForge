# coding=utf-8
"""The capture-server probe reports what an online producer would see."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DOMINO_DRAFT = os.path.join(REPO_ROOT, "configs", "qwen3-8b-domino.json")
HIDDEN = 8


def _target_dir(root: str) -> str:
    path = os.path.join(root, "target")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_type": "qwen3",
                "hidden_size": HIDDEN,
                "intermediate_size": 2 * HIDDEN,
                "num_hidden_layers": 40,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-5,
                "tie_word_embeddings": False,
            },
            handle,
        )
    return path


def _aux_layers() -> tuple:
    with open(DOMINO_DRAFT, encoding="utf-8") as handle:
        payload = json.load(handle)
    return tuple(payload["dflash_config"]["target_layer_ids"])


def _server(*, with_last_hidden: bool):
    """Emulate the patched server's meta_info result for one probe request."""

    layers = _aux_layers()

    def post(url: str, json_body: dict, timeout: float):
        rows = []
        for input_ids, spec in zip(json_body["input_ids"], json_body["spec_capture"]):
            length = len(input_ids)
            feats = {}
            mapping = spec["features"]
            if "aux" in mapping:
                feats[mapping["aux"]] = {
                    "shape": [1, length, len(layers) * HIDDEN],
                    "dtype": "bfloat16",
                }
            if "last_hidden" in mapping and with_last_hidden:
                feats[mapping["last_hidden"]] = {
                    "shape": [1, length, HIDDEN],
                    "dtype": "bfloat16",
                }
            for item in spec.get("passthrough", ()):
                feats[item["name"]] = {"shape": list(item["shape"]), "dtype": "int64"}
            rows.append(
                {
                    "meta_info": {
                        "spec_capture": {
                            "sample_id": spec["sample_id"],
                            "store_id": spec["store_id"],
                            "gen": int(spec["gen"]),
                            "aux_layer_ids": list(layers),
                            "features": feats,
                        }
                    }
                }
            )
        return rows

    return post


def _run_probe(root: str, *, args, post):
    from scripts import probe_capture_server

    out = io.StringIO()
    with mock.patch(
        "specforge.inference.adapters.server_capture._default_post",
        side_effect=post,
    ):
        with contextlib.redirect_stdout(out):
            code = probe_capture_server.main(
                [
                    "--server-url",
                    "http://server:30000",
                    "--target-model-path",
                    _target_dir(root),
                    "--draft-model-config",
                    DOMINO_DRAFT,
                    *args,
                ]
            )
    return code, out.getvalue()


class CaptureProbeTest(unittest.TestCase):
    def test_ce_only_probe_passes_against_a_server_without_last_hidden(self):
        """The probe must mirror the run: CE-only needs no teacher artifact."""

        with tempfile.TemporaryDirectory() as root:
            code, text = _run_probe(
                root,
                args=[],
                post=_server(with_last_hidden=False),
            )

        self.assertEqual(0, code, text)
        self.assertIn("PROBE OK", text)
        self.assertIn("last_hidden artifact: None", text)
        self.assertIn("hidden_states", text)

    def test_l1_probe_passes_when_the_server_returns_last_hidden(self):
        with tempfile.TemporaryDirectory() as root:
            code, text = _run_probe(
                root,
                args=["--l1"],
                post=_server(with_last_hidden=True),
            )

        self.assertEqual(0, code, text)
        self.assertIn("target_last_hidden_states", text)

    def test_l1_probe_reports_the_missing_teacher_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            code, text = _run_probe(
                root,
                args=["--l1"],
                post=_server(with_last_hidden=False),
            )

        self.assertEqual(1, code, text)
        self.assertIn("PROBE FAILED", text)
        self.assertIn("target_last_hidden_states", text)
        self.assertIn("re-apply the patch", text)

    def test_probe_reports_an_unpatched_server(self):
        def post(url, json_body, timeout):
            return [{"meta_info": {}} for _ in json_body["input_ids"]]

        with tempfile.TemporaryDirectory() as root:
            code, text = _run_probe(root, args=[], post=post)

        self.assertEqual(1, code, text)
        self.assertIn("no spec_capture result", text)
        self.assertIn("--enable-spec-capture", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
