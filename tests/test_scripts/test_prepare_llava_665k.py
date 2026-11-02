import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).parent.parent.parent

import datasets
from datasets import Features, Image as ImageFeature, Value
from PIL import Image as PILImage


def _jpeg(color):
    img = PILImage.new("RGB", (8, 8), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _shard(path, rows):
    texts_spec = [{"user": Value("string"), "assistant": Value("string")}]
    features = Features(
        {
            "id": Value("string"),
            "subset": Value("string"),
            "images": ImageFeature(),
            "texts": texts_spec,
        }
    )
    ds = datasets.Dataset.from_list(rows, features=features)
    ds.save_to_disk(str(path))
    return path / "data-00000-of-00001.arrow"


class TestPrepareLlava665k(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.base = Path(self.tmp.name)
        red = _jpeg((255, 0, 0))
        self.red = red
        rows = [
            {
                "id": "a",
                "subset": "coco",
                "images": {"bytes": red, "path": None},
                "texts": [{"user": "u1", "assistant": "a1"}],
            },
            {
                "id": "b",
                "subset": "coco",
                "images": {"bytes": red, "path": None},
                "texts": [
                    {"user": "u2", "assistant": "a2"},
                    {"user": "u3", "assistant": "a3"},
                ],
            },
            {
                "id": "c",
                "subset": "text",
                "images": None,
                "texts": [{"user": "u4", "assistant": "a4"}],
            },
        ]
        self.shard = _shard(self.base / "ds", rows)
        self.out = self.base / "out.jsonl"
        self.images = self.base / "images"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *extra):
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/prepare_llava_665k.py",
                "--shards",
                str(self.shard),
                "--output",
                str(self.out),
                "--image-dir",
                str(self.images),
                *extra,
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)
        return proc.stdout

    def test_convert_rows_and_dedupe_images(self):
        report = self._run()
        rows = [json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(["a", "b", "c"], [r["id"] for r in rows])
        # two identical image rows -> one file on disk, both records reference it
        self.assertEqual(1, len(rows[0]["images"]))
        self.assertEqual(rows[0]["images"], rows[1]["images"])
        self.assertEqual([], rows[2]["images"])
        files = list(self.images.rglob("*.jpg"))
        self.assertEqual(1, len(files))
        self.assertEqual(self.red, files[0].read_bytes())
        self.assertEqual("coco", rows[0]["subset"])
        self.assertEqual(
            [
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "u3"},
                {"role": "assistant", "content": "a3"},
            ],
            rows[1]["conversations"],
        )
        self.assertIn("multi-turn (>=2)     : 1 (33.3%)", report)

    def test_rerun_reuses_files(self):
        self._run()
        first = list(self.images.rglob("*.jpg"))[0]
        mtime = first.stat().st_mtime_ns
        report = self._run()
        again = list(self.images.rglob("*.jpg"))[0]
        self.assertEqual(mtime, again.stat().st_mtime_ns)
        self.assertIn("reused on disk     : 2", report)

    def test_subset_filter(self):
        self._run("--subsets", "text")
        rows = [json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(["c"], [r["id"] for r in rows])
        self.assertEqual([], list(self.images.rglob("*")))

    def test_num_samples_limit(self):
        self._run("--num-samples", "1")
        rows = [json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(1, len(rows))


if __name__ == "__main__":
    unittest.main()
