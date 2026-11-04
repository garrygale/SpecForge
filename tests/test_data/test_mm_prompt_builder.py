# coding=utf-8
"""Unit tests for the mixed text/image prompt pipeline (no network, fakes only).

The fake tokenizer is char-level (``id = ord(c) % 1000``) with the image
placeholder as one special token id -- mirroring real VLM tokenizers -- so
prefix re-encoding is exactly consistent. The fake processor expands one
``IMG`` placeholder into IMG_GRID copies of that id, mirroring Qwen-VL style
string-level expansion.
"""

import gc
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from specforge.algorithms.common import image_input
from specforge.algorithms.common.image_input import (
    ImageTextAdapter,
    image_streaming_adapter,
    mm_prompt_cache_key,
)

IMG_ID = 7331
IMG_GRID = 4


def _char_ids(text):
    ids, i = [], 0
    while i < len(text):
        if text.startswith("IMG", i):
            ids.append(IMG_ID)
            i += 3
        else:
            ids.append(ord(text[i]) % 1000)
            i += 1
    return ids


class FakeTokenizer:
    pad_token_id = 0
    unk_token_id = 0
    chat_template = None

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        parts = []
        for m in messages:
            parts.append(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            )
        return "".join(parts)

    def encode(self, text, add_special_tokens=False, **kwargs):
        ids = _char_ids(text)
        max_length = kwargs.get("max_length")
        return ids[:max_length] if max_length else ids

    def convert_tokens_to_ids(self, token):
        return IMG_ID if token == "IMG" else None

    def __call__(
        self,
        text,
        max_length=None,
        truncation=False,
        return_tensors=None,
        add_special_tokens=False,
    ):
        ids = _char_ids(text)
        if truncation and max_length:
            ids = ids[:max_length]
        return SimpleNamespace(input_ids=torch.tensor([ids]))


class FakeProcessor:
    image_token = "IMG"

    def __call__(self, text, images, **kwargs):
        if not isinstance(text, str):
            text = text[0]
        ids = []
        i = 0
        while i < len(text):
            if text.startswith(self.image_token, i):
                ids.extend([IMG_ID] * IMG_GRID)
                i += len(self.image_token)
            else:
                ids.append(ord(text[i]) % 1000)
                i += 1
        return {"input_ids": [ids]}


def _mm_row(id_, user, assistant, image=None):
    row = {
        "id": id_,
        "images": [image] if image else [],
        "conversations": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }
    return row


class TestMMPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.image_rel = "img0.png"
        Image.new("RGB", (4, 4), (0, 255, 0)).save(
            os.path.join(self.root, self.image_rel)
        )
        self.cache_dir = os.path.join(self.root, "cache")
        self.tokenizer = FakeTokenizer()
        self.processor = FakeProcessor()
        self._load_processor_patch = patch.object(
            image_input, "load_processor", lambda config: self.processor
        )
        self._load_processor_patch.start()

    def tearDown(self):
        self._load_processor_patch.stop()
        # HF datasets keeps the arrow cache mmap'ed; on Windows the lock can
        # outlive the objects, so cleanup is best-effort.
        gc.collect()
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass

    def _config(self, jsonl_name="data.jsonl", **overrides):
        path = os.path.join(self.root, jsonl_name)
        data = dict(
            train_data_path=path,
            prompts_path="",
            image_root=self.root,
            chat_template="qwen",
            max_length=256,
            is_preformatted=False,
            train_only_last_turn=False,
            max_prompts=None,
            cache_dir=self.cache_dir,
            cache_key=None,
            build_dataset_num_proc=1,
            image_min_pixels=None,
            image_max_pixels=None,
        )
        data.update(overrides)
        return SimpleNamespace(
            data=SimpleNamespace(**data),
            model=SimpleNamespace(
                target_model_path="fake/vlm",
                cache_dir=None,
                trust_remote_code=False,
                tokenizer_pad_token_id=0,
            ),
        )

    def _write_rows(self, rows, name="data.jsonl"):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return path

    def test_mixed_dataset_expands_image_and_masks_assistant_spans(self):
        rows = [
            _mm_row("mm", "describe the bus", "red", image=self.image_rel),
            _mm_row("txt", "plain question", "plain answer"),
        ]
        self._write_rows(rows)
        adapter = ImageTextAdapter()
        prompts = adapter.prepare_prompts(
            self._config(), self.tokenizer, draft_config=None
        )
        self.assertEqual(2, len(prompts))
        by_image = {bool(p["payload"]["image"]): p["payload"] for p in prompts}

        mm = by_image[True]
        self.assertEqual(self.image_rel, mm["image"])
        self.assertIn("IMG", mm["prompt_text"])
        ids = mm["input_ids"]
        mask = mm["loss_mask"]
        self.assertEqual(len(ids), len(mask))
        # the placeholder was expanded into the image token grid, and the
        # grid is prompt-side (masked out)
        self.assertEqual(IMG_GRID, ids.count(IMG_ID))
        for token, flagged in zip(ids, mask):
            if token == IMG_ID:
                self.assertEqual(0, flagged)
        # char-level tokenizer: the assistant span plus the end-of-turn
        # token is supervised (the model must learn to emit it)
        eot = len("<|im_end|>\n")
        self.assertEqual(len("red") + eot, sum(mask))

        text = by_image[False]
        self.assertIsNone(text["image"])
        self.assertIsNone(text["prompt_text"])
        self.assertEqual(len("plain answer") + eot, sum(text["loss_mask"]))
        self.assertEqual(
            [ord(c) % 1000 for c in "plain answer"],
            text["input_ids"][-(len("plain answer") + eot) : -eot],
        )

    def test_multi_turn_image_conversation_masks_every_assistant_turn(self):
        rows = [
            {
                "id": "multi",
                "images": [self.image_rel],
                "conversations": [
                    {"role": "user", "content": "what is this"},
                    {"role": "assistant", "content": "a bus"},
                    {"role": "user", "content": "what color"},
                    {"role": "assistant", "content": "white and red"},
                ],
            }
        ]
        self._write_rows(rows)
        prompts = ImageTextAdapter().prepare_prompts(
            self._config(), self.tokenizer, draft_config=None
        )
        payload = prompts[0]["payload"]
        self.assertEqual(
            len("a bus") + len("white and red") + 2 * len("<|im_end|>\n"),
            sum(payload["loss_mask"]),
        )

    def test_cache_stores_string_columns_and_no_image_bytes(self):
        self._write_rows([_mm_row("mm", "u", "a", image=self.image_rel)])
        cfg = self._config()
        prompts = ImageTextAdapter().prepare_prompts(
            cfg, self.tokenizer, draft_config=None
        )
        processed = prompts._dataset
        features = processed.features
        self.assertEqual("string", str(features["image"].dtype))
        self.assertEqual("string", str(features["prompt_text"].dtype))
        self.assertIn("input_ids", features)
        self.assertIn("loss_mask", features)

        # second prepare_prompts reads the same cache file and round-trips
        again = ImageTextAdapter().prepare_prompts(
            cfg, self.tokenizer, draft_config=None
        )
        self.assertEqual(prompts[0]["payload"], again[0]["payload"])
        cache_files = [
            name
            for name in os.listdir(self.cache_dir)
            if name.endswith(".pkl") or name.endswith(".arrow")
        ]
        self.assertTrue(cache_files)
        for name in cache_files:
            size = os.path.getsize(os.path.join(self.cache_dir, name))
            self.assertLess(size, 200_000, f"{name} grew beyond token-only size")

    def test_multi_image_row_is_rejected(self):
        row = _mm_row("two", "u", "a", image=self.image_rel)
        row["images"].append(self.image_rel)
        self._write_rows([row])
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            ImageTextAdapter().prepare_prompts(
                self._config(), self.tokenizer, draft_config=None
            )

    def test_min_loss_tokens_fn_drops_rows_without_trainable_tokens(self):
        rows = [
            _mm_row("keep", "u", "real answer", image=self.image_rel),
            {"id": "drop", "images": [], "conversations": [
                {"role": "user", "content": "only question"},
                {"role": "assistant", "content": "x"},
            ]},
        ]
        self._write_rows(rows)
        # trainable span = content + end-of-turn token, so thresholds are
        # 22 (keep) vs 12 (drop)
        adapter = ImageTextAdapter(minimum_loss_tokens_fn=lambda cfg, dc: 13)
        prompts = adapter.prepare_prompts(
            self._config(), self.tokenizer, draft_config=None
        )
        # the 1-token answer row is dropped, the 11-char one survives
        self.assertEqual(1, len(prompts))
        self.assertTrue(prompts[0]["payload"]["image"])

    def test_mm_cache_key_sensitivity(self):
        self._write_rows([_mm_row("mm", "u", "a", image=self.image_rel)])
        cfg = self._config()
        other_root = self._config(image_root=os.path.join(self.root, "elsewhere"))
        self.assertNotEqual(
            mm_prompt_cache_key(cfg, tokenizer=self.tokenizer, processor=self.processor),
            mm_prompt_cache_key(
                other_root, tokenizer=self.tokenizer, processor=self.processor
            ),
        )
        self.assertEqual(
            mm_prompt_cache_key(cfg, tokenizer=self.tokenizer, processor=self.processor),
            mm_prompt_cache_key(cfg, tokenizer=self.tokenizer, processor=self.processor),
        )


class TestBuildRequestInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        with open(os.path.join(self.root, "img.jpg"), "wb") as fh:
            fh.write(b"\xff\xd8jpeg-bytes")
        factory = image_streaming_adapter()
        self.adapter = factory(
            SimpleNamespace(
                data=SimpleNamespace(
                    image_root=self.root,
                    image_min_pixels=None,
                    image_max_pixels=None,
                )
            )
        )
        self.adapter._image_token_id = IMG_ID

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _task(name, image=None, prompt_text=None, ids=None):
        return SimpleNamespace(
            payload={
                "input_ids": ids if ids is not None else [1, 2, 3],
                "loss_mask": [0, 1, 1],
                "image": image,
                "prompt_text": prompt_text,
            }
        )

    def test_split_request_groups_is_a_single_group(self):
        text0 = self._task("t0")
        image1 = self._task("t1", image="img.jpg", prompt_text="p1")
        text2 = self._task("t2")
        groups = self.adapter.split_request_groups([text0, image1, text2])
        self.assertEqual([[text0, image1, text2]], groups)

    def test_build_request_inputs_unified_input_ids(self):
        image = self._task("t1", image="img.jpg", prompt_text="rendered", ids=[9, 8])
        text = self._task("t0", ids=[1, 2, 3])
        inputs = self.adapter.build_request_inputs([text, image])
        self.assertEqual([[1, 2, 3], [9, 8]], inputs["input_ids"])
        self.assertEqual(2, len(inputs["image_data"]))
        self.assertIsNone(inputs["image_data"][0])
        (uri,) = inputs["image_data"][1]
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        import base64

        self.assertEqual(
            b"\xff\xd8jpeg-bytes", base64.b64decode(uri.split(",", 1)[1])
        )
        self.assertNotIn("text", inputs)

    def test_build_request_inputs_text_only_omits_image_data(self):
        text = self._task("t0", ids=[1, 2, 3])
        inputs = self.adapter.build_request_inputs([text])
        self.assertEqual([[1, 2, 3]], inputs["input_ids"])
        self.assertNotIn("image_data", inputs)


class FileCountingProcessor(FakeProcessor):
    """FakeProcessor that logs every call to a file (spawn-safe counting)."""

    def __init__(self, counter_path):
        self.counter_path = counter_path

    def __call__(self, text, images, **kwargs):
        with open(self.counter_path, "a", encoding="utf-8") as fh:
            fh.write("1")
        return super().__call__(text, images, **kwargs)


class NonStringLevelProcessor(FakeTokenizer):
    """Processor whose expansion is NOT the string-level form: the expanded
    ids disagree with tokenizing the placeholder-repeated text, forcing the
    per-prefix processor fallback."""

    image_token = "IMG"

    def __call__(self, text, images, **kwargs):
        t = text if isinstance(text, str) else text[0]
        ids = [9900] + [ord(c) % 1000 for c in t.replace("IMG", "", 1)]
        return {"input_ids": [ids]}


class TestProcessorCallBudget(unittest.TestCase):
    """The map must run full image preprocessing once per unique image."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        Image.new("RGB", (4, 4), (0, 0, 255)).save(
            os.path.join(self.root, "shared.png")
        )
        self.counter = os.path.join(self.root, "calls.txt")
        open(self.counter, "w").close()
        self.processor = FileCountingProcessor(self.counter)
        self._patch = patch.object(
            image_input, "load_processor", lambda config: self.processor
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        gc.collect()
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass

    def _config(self):
        return SimpleNamespace(
            data=SimpleNamespace(
                train_data_path=os.path.join(self.root, "rows.jsonl"),
                prompts_path="",
                image_root=self.root,
                chat_template="qwen",
                max_length=256,
                is_preformatted=False,
                train_only_last_turn=False,
                max_prompts=None,
                cache_dir=os.path.join(self.root, "cache"),
                cache_key=None,
                build_dataset_num_proc=1,
                image_min_pixels=None,
                image_max_pixels=None,
            ),
            model=SimpleNamespace(
                target_model_path="fake/vlm",
                cache_dir=None,
                trust_remote_code=False,
                tokenizer_pad_token_id=0,
            ),
        )

    def _calls(self):
        with open(self.counter, encoding="utf-8") as fh:
            return len(fh.read())

    def test_one_processor_call_per_unique_image(self):
        rows = [
            _mm_row("a", "question one", "answer one", image="shared.png"),
            _mm_row("b", "question two", "answer two", image="shared.png"),
            _mm_row("c", "question three", "answer three", image="shared.png"),
            _mm_row("t", "plain", "words", image=None),
        ]
        with open(os.path.join(self.root, "rows.jsonl"), "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        prompts = ImageTextAdapter().prepare_prompts(
            self._config(), FakeTokenizer(), draft_config=None
        )
        self.assertEqual(4, len(prompts))
        eot = len("<|im_end|>\n")
        for prompt in prompts:
            payload = prompt["payload"]
            if payload["image"]:
                content = (
                    "answer one"
                    if "one" in payload["prompt_text"]
                    else "answer two"
                    if "two" in payload["prompt_text"]
                    else "answer three"
                )
                self.assertEqual(len(content) + eot, sum(payload["loss_mask"]))
        # exactly one full-preprocessing call for the single shared image
        self.assertEqual(1, self._calls())

    def test_non_string_level_expansion_falls_back_to_processor_prefixes(self):
        fallback = patch.object(
            image_input, "load_processor", lambda config: NonStringLevelProcessor()
        )
        with fallback:
            rows = [_mm_row("x", "q", "resp", image="shared.png")]
            with open(
                os.path.join(self.root, "rows.jsonl"), "w", encoding="utf-8"
            ) as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            prompts = ImageTextAdapter().prepare_prompts(
                self._config(), FakeTokenizer(), draft_config=None
            )
        payload = prompts[0]["payload"]
        # slow path: full call + one prefix pair per assistant span
        self.assertEqual(len("resp") + len("<|im_end|>\n"), sum(payload["loss_mask"]))
        self.assertIn("IMG", payload["prompt_text"])


class BoundedFakeProcessor(FakeProcessor):
    """FakeProcessor with a Qwen-style image_processor pixel config."""

    def __init__(self):
        self.image_processor = SimpleNamespace(
            min_pixels=4 * 28 * 28, max_pixels=16384 * 28 * 28
        )


class TestPixelBoundsAndPlaceholderForm(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        with open(os.path.join(self.root, "img.png"), "wb") as fh:
            fh.write(b"\x89PNG-bytes")

    def tearDown(self):
        gc.collect()
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass

    def _adapter(self):
        adapter = image_streaming_adapter()(
            SimpleNamespace(
                data=SimpleNamespace(
                    image_root=self.root,
                    image_min_pixels=None,
                    image_max_pixels=None,
                )
            )
        )
        adapter._image_token_id = IMG_ID
        return adapter

    def test_pixel_bounds_applied_and_keyed(self):
        processor = BoundedFakeProcessor()
        with patch.object(image_input, "load_processor", lambda config: processor):
            adapter = ImageTextAdapter()
            config = SimpleNamespace(
                data=SimpleNamespace(
                    image_min_pixels=56 * 56,
                    image_max_pixels=1280 * 28 * 28,
                )
            )
            from specforge.algorithms.common.image_input import _apply_pixel_bounds

            _apply_pixel_bounds(processor, config)
        self.assertEqual(56 * 56, processor.image_processor.min_pixels)
        self.assertEqual(1280 * 28 * 28, processor.image_processor.max_pixels)

        # a processor without pixel support fails loud when bounds are set
        plain = FakeProcessor()
        with self.assertRaisesRegex(ValueError, "does not expose"):
            _apply_pixel_bounds(
                plain,
                SimpleNamespace(
                    data=SimpleNamespace(
                        image_min_pixels=1, image_max_pixels=None
                    )
                ),
            )

    def test_pixel_bounds_change_cache_key(self):
        self._write_one_row()
        base = SimpleNamespace(
            data=SimpleNamespace(
                train_data_path=os.path.join(self.root, "row.jsonl"),
                prompts_path="",
                image_root=self.root,
                chat_template="qwen",
                max_length=256,
                is_preformatted=False,
                train_only_last_turn=False,
                max_prompts=None,
                image_min_pixels=None,
                image_max_pixels=None,
            ),
            model=SimpleNamespace(target_model_path="fake/vlm"),
        )
        tok = FakeTokenizer()
        proc = FakeProcessor()
        key = mm_prompt_cache_key(base, tokenizer=tok, processor=proc)
        base.data.image_max_pixels = 1280 * 28 * 28
        self.assertNotEqual(
            key, mm_prompt_cache_key(base, tokenizer=tok, processor=proc)
        )

    def test_request_collapses_grid_to_one_placeholder(self):
        adapter = self._adapter()
        image = SimpleNamespace(
            payload={
                "input_ids": [1, IMG_ID, IMG_ID, IMG_ID, IMG_ID, 2, 3],
                "loss_mask": [0, 0, 0, 0, 0, 1, 1],
                "image": "img.png",
                "prompt_text": "unused-at-request-time",
            }
        )
        inputs = adapter.build_request_inputs([image])
        self.assertEqual([[1, IMG_ID, 2, 3]], inputs["input_ids"])
        (uri,) = inputs["image_data"][0]
        self.assertTrue(uri.startswith("data:image/png;base64,"))

        text = SimpleNamespace(
            payload={
                "input_ids": [1, IMG_ID, 2],
                "loss_mask": [0, 1, 1],
                "image": None,
                "prompt_text": None,
            }
        )
        inputs = adapter.build_request_inputs([text])
        self.assertNotIn("image_data", inputs)

    def _write_one_row(self):
        from PIL import Image as PILImage

        PILImage.new("RGB", (4, 4)).save(os.path.join(self.root, "img.png"))
        with open(os.path.join(self.root, "row.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(_mm_row("a", "q", "a", image="img.png")))


if __name__ == "__main__":
    unittest.main()
