import base64
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import regen_responses_mm
from scripts.regen_responses_mm import (
    _has_complete_responses,
    _image_data_uri,
    _strip_image_marker,
    _to_openai_messages,
    regenerate_one,
)


class _Handler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible mock: /v1/chat/completions + /tokenize.

    The reply content is ``resp<turn>`` (turn = assistant messages already in
    the history), so tests can assert sequential regeneration. Requests whose
    last user text contains a marker get special behaviour:

    ``FORCE_TRUNC`` -> finish_reason "length"; ``FORCE_EMPTY`` -> null content.
    Every request is recorded; image parts are decoded and checked.
    """

    requests = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        if self.path.endswith("/tokenize"):
            self._json({"count": max(1, len(body["prompt"]) // 10)})
            return
        if self.path.endswith("/chat/completions"):
            _Handler.requests.append(body)
            messages = body["messages"]
            turn = sum(1 for m in messages if m["role"] == "assistant")
            last_user = next(
                m for m in reversed(messages) if m["role"] == "user"
            )
            last_text = last_user["content"] if isinstance(last_user["content"], str) else next(
                part["text"] for part in last_user["content"] if part["type"] == "text"
            )
            content = f"resp{turn}"
            finish = "stop"
            if "FORCE_TRUNC" in last_text:
                finish = "length"
            elif "FORCE_EMPTY" in last_text:
                content = None
            self._json(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", "default"),
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": finish,
                            "message": {"role": "assistant", "content": content},
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )

    def _json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def _image_parts(request):
    parts = []
    for message in request["messages"]:
        if isinstance(message.get("content"), list):
            parts.extend(p for p in message["content"] if p["type"] == "image_url")
    return parts


class TestRegenResponsesMm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        _Handler.requests = []
        self.tmp = TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.jpg = self.base / "img.jpg"
        self.jpg.write_bytes(b"\xff\xd8fake-jpeg-bytes")

    def tearDown(self):
        self.tmp.cleanup()

    def _regenerate(self, sample, **overrides):
        kwargs = dict(
            vllm_url=self.url,
            model="default",
            temperature=0.7,
            max_tokens=64,
            top_k=0,
            top_p=1.0,
            enable_reasoning=False,
            reasoning_effort="medium",
            max_reasoning_tokens=None,
            feed_reasoning=False,
            timeout=30.0,
            image_root=str(self.base),
        )
        kwargs.update(overrides)
        return regenerate_one(sample, **kwargs)

    def test_data_uri_and_marker(self):
        uri = _image_data_uri("img.jpg", str(self.base))
        self.assertEqual("data:image/jpeg;base64,", uri[:23])
        self.assertEqual(
            b"\xff\xd8fake-jpeg-bytes",
            base64.b64decode(uri.split(",", 1)[1]),
        )
        (self.base / "img.png").write_bytes(b"png")
        self.assertEqual("data:image/png;base64,", _image_data_uri("img.png", str(self.base))[:22])
        self.assertEqual("What is this?", _strip_image_marker("<image>\nWhat is this?"))
        self.assertEqual("What?", _strip_image_marker("<image>What?"))
        self.assertEqual("plain", _strip_image_marker("plain"))

    def test_openai_messages_attach_images_to_first_user_turn_only(self):
        conversations = [
            {"role": "user", "content": "<image>\ndescribe"},
            {"role": "assistant", "content": "old"},
            {"role": "user", "content": "again"},
        ]
        messages = _to_openai_messages(conversations, image_uris=["data:image/jpeg;base64,QUJD"])
        self.assertEqual(
            [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
                {"type": "text", "text": "describe"},
            ],
            messages[0]["content"],
        )
        self.assertEqual("again", messages[2]["content"])
        plain = _to_openai_messages(conversations)
        self.assertTrue(all(isinstance(m["content"], str) for m in plain))

    def test_regenerate_one_multimodal(self):
        sample = {
            "id": "x",
            "images": ["img.jpg"],
            "conversations": [
                {"role": "user", "content": "describe"},
                {"role": "assistant", "content": "old1"},
                {"role": "user", "content": "again"},
            ],
        }
        result = self._regenerate(sample)
        self.assertEqual("success", result["status"])
        self.assertEqual(
            [
                {"role": "user", "content": "describe"},
                {"role": "assistant", "content": "resp0"},
                {"role": "user", "content": "again"},
                {"role": "assistant", "content": "resp1"},
            ],
            result["conversations"],
        )
        self.assertEqual(["img.jpg"], result["images"])
        self.assertEqual(2, len(_Handler.requests))
        first, second = _Handler.requests
        first_parts = _image_parts(first)
        self.assertEqual(1, len(first_parts))
        self.assertEqual(
            b"\xff\xd8fake-jpeg-bytes",
            base64.b64decode(first_parts[0]["image_url"]["url"].split(",", 1)[1]),
        )
        # every request's history re-sends the image on the first user turn
        # (the server re-prefills the whole conversation each time), while the
        # new user turn itself stays plain text
        self.assertEqual(1, len(_image_parts(second)))
        self.assertEqual(
            "image_url", _image_parts(second)[0]["type"]
        )
        last_second = second["messages"][-1]
        self.assertEqual("user", last_second["role"])
        self.assertEqual("again", last_second["content"])

    def test_regenerate_one_text_only(self):
        sample = {
            "id": "t",
            "images": [],
            "conversations": [{"role": "user", "content": "hi"}],
        }
        result = self._regenerate(sample)
        self.assertEqual("success", result["status"])
        self.assertEqual([{"role": "assistant", "content": "resp0"}], result["conversations"][1:])
        self.assertEqual(0, len(_image_parts(_Handler.requests[0])))

    def test_truncated_flag_and_empty_content(self):
        truncated = self._regenerate(
            {"conversations": [{"role": "user", "content": "FORCE_TRUNC"}]}
        )
        self.assertEqual("success", truncated["status"])
        self.assertIs(True, truncated["conversations"][1]["truncated"])

        empty = self._regenerate(
            {"conversations": [{"role": "user", "content": "FORCE_EMPTY"}]}
        )
        self.assertEqual("error", empty["status"])
        self.assertIn("empty content", empty["error"])

    def test_missing_image_is_sample_error(self):
        result = self._regenerate(
            {"images": ["gone.jpg"], "conversations": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual("error", result["status"])
        self.assertIn("Cannot read image", result["error"])

    def test_has_complete_responses(self):
        self.assertFalse(_has_complete_responses([{"role": "user", "content": "u"}]))
        self.assertTrue(
            _has_complete_responses(
                [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
            )
        )

    def test_main_retry_over_tokens_reads_state_from_input(self):
        input_path = self.base / "in.jsonl"
        output_dir = self.base / "out"
        rows = [
            # kept: complete, short, not truncated
            {
                "id": "keep",
                "images": [],
                "conversations": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "short answer"},
                ],
            },
            # regenerated: truncated flag
            {
                "id": "trunc",
                "images": ["img.jpg"],
                "conversations": [
                    {"role": "user", "content": "FORCE_TRUNC"},
                    {"role": "assistant", "content": "cut off", "truncated": True},
                ],
            },
            # regenerated: response over the token threshold
            {
                "id": "long",
                "images": [],
                "conversations": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "word " * 80},
                ],
            },
            # regenerated: freshly converted input (user turn without response)
            {
                "id": "fresh",
                "images": [],
                "conversations": [{"role": "user", "content": "u"}],
            },
        ]
        input_path.write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        with patch(
            "sys.argv",
            [
                "regen_responses_mm.py",
                "--input",
                str(input_path),
                "--image-dir",
                str(self.base),
                "--vllm-url",
                self.url,
                "--output-dir",
                str(output_dir),
                "--retry-over-tokens",
                "5",
                "--num-workers",
                "2",
            ],
        ):
            regen_responses_mm.main()

        out_rows = [
            json.loads(line)
            for line in (output_dir / "in_regen.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        by_id = {r["id"]: r for r in out_rows}
        self.assertEqual({"keep", "trunc", "long", "fresh"}, set(by_id))
        # kept sample is verbatim, original response preserved
        self.assertEqual("short answer", by_id["keep"]["conversations"][1]["content"])
        self.assertEqual("success", by_id["keep"]["status"])
        # regenerated samples carry mock responses; images survive the round-trip
        self.assertEqual("resp0", by_id["trunc"]["conversations"][1]["content"])
        self.assertEqual(["img.jpg"], by_id["trunc"]["images"])
        self.assertEqual("resp0", by_id["long"]["conversations"][1]["content"])
        self.assertEqual("resp0", by_id["fresh"]["conversations"][1]["content"])


if __name__ == "__main__":
    unittest.main()
