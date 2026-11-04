"""Mixed text/image prompt dataset building for the image input adapter.

Rows without images go through the exact text pipeline
(:func:`preprocess_conversations`). Rows with an image render through the
same parser/chat template, then the multimodal processor expands the image
placeholder token into the real per-image token grid (Qwen-VL style), so
``input_ids`` matches token-for-token what the capture server produces for
the same text + image.

The map cache stores only token tensors plus two string columns (``image``
path, ``prompt_text``) -- image bytes are read from disk during the map and
again at request time, never serialized into the cache.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch

from .template import TEMPLATE_REGISTRY, ChatTemplate


def _image_ids(processor, text: str, images) -> list:
    out = processor(text=text, images=images)
    ids = out["input_ids"] if hasattr(out, "__getitem__") else out.input_ids
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def _text_ids(tokenizer, text: str) -> list:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if torch.is_tensor(ids):
        ids = ids.tolist()
    return [int(t) for t in ids]


def _spans_to_mask(spans, tokenizer, text: str, ids_len: int, max_length: int):
    """Turn (start_char, end_char) assistant spans into a token loss mask."""
    loss_mask = torch.zeros(1, ids_len, dtype=torch.long)
    for start_char, end_char in spans:
        start_idx = min(len(_text_ids(tokenizer, text[:start_char])[:max_length]), ids_len)
        end_idx = min(len(_text_ids(tokenizer, text[:end_char])[:max_length]), ids_len)
        if start_idx < end_idx:
            loss_mask[0, start_idx:end_idx] = 1
    return loss_mask


def _mask_from_spans_via_processor(spans, processor, images, text, ids_len, max_length):
    """Same as _spans_to_mask but every prefix runs the full processor."""
    loss_mask = torch.zeros(1, ids_len, dtype=torch.long)
    for start_char, end_char in spans:
        start_idx = min(
            len(_image_ids(processor, text[:start_char], images)[:max_length]), ids_len
        )
        end_idx = min(
            len(_image_ids(processor, text[:end_char], images)[:max_length]), ids_len
        )
        if start_idx < end_idx:
            loss_mask[0, start_idx:end_idx] = 1
    return loss_mask


def _prepare_messages(parser, conversation) -> list:
    """Mirror GeneralParser.parse's message prep (normalize/system/sanitize)."""
    conversation = [parser._normalize_message(message) for message in conversation]
    messages = []
    if conversation and conversation[0]["role"] == "system":
        messages.append({"role": "system", "content": conversation[0]["content"]})
        conversation = conversation[1:]
    elif parser.system_prompt:
        messages.append({"role": "system", "content": parser.system_prompt})
    while conversation and conversation[0]["role"] != "user":
        conversation = conversation[1:]
    for j, sentence in enumerate(conversation):
        role = sentence["role"]
        if j == 0 and role != "user":
            break
        if j > 0:
            prev_role = conversation[j - 1]["role"]
            if role == "tool" and prev_role not in ("assistant", "tool"):
                break
            if role == "assistant" and prev_role not in ("user", "tool"):
                break
        messages.append(parser._sanitize_message(sentence))
    return messages


def encode_image_conversation(
    parser,
    processor,
    image_paths: list,
    image_root: str,
    conversation,
    max_length: int,
    train_only_last_turn: bool = False,
    tool=None,
    image_placeholder: str = "<|image_pad|>",
    image_token: str = "<|image_pad|>",
    n_cache: Optional[dict] = None,
) -> tuple:
    """Render + expand one image-grounded conversation.

    Returns ``(input_ids[1, L], loss_mask[1, L], rendered_text)`` where the
    rendered text keeps the single image placeholder (the capture server
    expands it itself) while the ids are processor-expanded so they align
    with the captured hidden states.

    Cost model: the processor runs its full image preprocessing ONCE per
    unique image path (memoized in ``n_cache``); everything else -- assistant
    span anchoring, ignore-token zeroing -- uses the plain text tokenizer.
    String-level expansion (processor replaces the placeholder with N copies
    of ``image_token`` and then tokenizes) holds for the standard HF
    multimodal processors and is verified per first encounter; a processor
    that violates it falls back to per-prefix processor calls for that
    sample.
    """
    from PIL import Image

    if len(image_paths) != 1:
        raise ValueError(
            f"expected exactly one image per sample, got {len(image_paths)}"
        )
    images = []
    for image_path in image_paths:
        full = (
            image_path
            if os.path.isabs(image_path)
            else os.path.join(image_root, image_path)
        )
        images.append(Image.open(full).convert("RGB"))

    messages = _prepare_messages(parser, conversation)
    if not any(m["role"] == "user" for m in messages):
        raise ValueError("conversation has no user turn after normalization")
    first_user = next(m for m in messages if m["role"] == "user")
    first_user["content"] = f"{image_placeholder}\n{first_user['content']}"

    rendered = parser.apply_chat_template(messages, tool=tool)

    matches = list(re.finditer(parser.assistant_pattern, rendered, re.DOTALL))
    if train_only_last_turn and matches:
        matches = [matches[-1]]
    spans = [(m.start(1), m.end(1)) for m in matches]
    ignore_spans = []
    for token_str in parser.chat_template.ignore_token or []:
        start = 0
        while True:
            idx = rendered.find(token_str, start)
            if idx == -1:
                break
            ignore_spans.append((idx, idx + len(token_str)))
            start = idx + len(token_str)

    expanded = _expand_once(
        parser,
        processor,
        image_paths[0],
        images,
        rendered,
        image_token,
        n_cache,
    )
    if expanded is None:
        return _encode_via_processor_prefixes(
            processor,
            images,
            rendered,
            spans,
            ignore_spans,
            max_length,
        )

    ids, expanded_text = expanded
    ids = ids[:max_length]
    input_ids = torch.tensor([ids], dtype=torch.long)
    expanded_spans = _shift_spans(spans, rendered, expanded_text, image_token)
    expanded_ignore = _shift_spans(ignore_spans, rendered, expanded_text, image_token)
    loss_mask = _spans_to_mask(expanded_spans, parser.tokenizer, expanded_text, len(ids), max_length)
    for start_char, end_char in expanded_ignore:
        start_idx = min(len(_text_ids(parser.tokenizer, expanded_text[:start_char])[:max_length]), len(ids))
        end_idx = min(len(_text_ids(parser.tokenizer, expanded_text[:end_char])[:max_length]), len(ids))
        if start_idx < end_idx:
            loss_mask[0, start_idx:end_idx] = 0
    return input_ids, loss_mask, rendered


def _expand_once(parser, processor, image_path, images, rendered, image_token, n_cache):
    """Processor-expanded ids for ``rendered``, or None when the string-level
    expansion assumption does not hold (caller falls back)."""
    if rendered.count(image_token) != 1:
        return None
    if n_cache is not None and image_path in n_cache:
        n = n_cache[image_path]
        ids = None
    else:
        ids = _image_ids(processor, rendered, images)
        base_len = len(_text_ids(parser.tokenizer, rendered))
        # the placeholder contributes c ids to the plain tokenizer and n to
        # the processor's expansion (n >= c); c is measured, not assumed
        c = len(_text_ids(parser.tokenizer, image_token))
        n = len(ids) - base_len + c
        if n < c:
            return None
    expanded_text = rendered.replace(image_token, image_token * n, 1)
    if ids is None:
        ids = _text_ids(parser.tokenizer, expanded_text)
    elif _text_ids(parser.tokenizer, expanded_text) != ids:
        return None
    if n_cache is not None and image_path not in n_cache:
        n_cache[image_path] = n
    return ids, expanded_text


def _shift_spans(spans, rendered, expanded_text, image_token):
    """Shift char spans past the placeholder by the expansion delta."""
    rendered_pos = rendered.index(image_token)
    delta = len(expanded_text) - len(rendered)
    shifted = []
    for start_char, end_char in spans:
        if start_char > rendered_pos:
            start_char += delta
            end_char += delta
        elif end_char > rendered_pos:
            return None
        shifted.append((start_char, end_char))
    return shifted


def _encode_via_processor_prefixes(processor, images, rendered, spans, ignore_spans, max_length):
    """Slow path: anchor every span with full processor prefix calls."""
    ids = _image_ids(processor, rendered, images)[:max_length]
    input_ids = torch.tensor([ids], dtype=torch.long)
    loss_mask = _mask_from_spans_via_processor(
        spans, processor, images, rendered, len(ids), max_length
    )
    for start_char, end_char in ignore_spans:
        start_idx = min(
            len(_image_ids(processor, rendered[:start_char], images)[:max_length]),
            len(ids),
        )
        end_idx = min(
            len(_image_ids(processor, rendered[:end_char], images)[:max_length]),
            len(ids),
        )
        if start_idx < end_idx:
            loss_mask[0, start_idx:end_idx] = 0
    return input_ids, loss_mask, rendered


class MMPromptSequence(Sequence):
    """Lazy payload view over the processed mixed dataset."""

    def __init__(
        self,
        dataset,
        *,
        max_length: int,
        min_loss_tokens: int,
        loss_mask_filter: Optional[Callable[[Sequence[int]], bool]] = None,
    ) -> None:
        self._dataset = dataset
        self._max_length = max_length
        self._min_loss_tokens = min_loss_tokens
        self._loss_mask_filter = loss_mask_filter

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        record = self._dataset[index]
        input_ids = record["input_ids"]
        loss_mask = record["loss_mask"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if hasattr(loss_mask, "tolist"):
            loss_mask = loss_mask.tolist()
        input_ids = input_ids[0] if input_ids and isinstance(input_ids[0], list) else input_ids
        loss_mask = loss_mask[0] if loss_mask and isinstance(loss_mask[0], list) else loss_mask
        input_ids = [int(t) for t in input_ids][: self._max_length]
        loss_mask = [int(t) for t in loss_mask][: self._max_length]
        if len(input_ids) != len(loss_mask):
            raise ValueError(
                f"processed dataset row {index} has mismatched "
                f"input_ids/loss_mask lengths"
            )
        if not input_ids:
            raise ValueError(f"processed dataset row {index} is empty")
        if sum(loss_mask) < self._min_loss_tokens:
            raise ValueError(
                f"processed dataset row {index} violates the preprocessing "
                f"minimum of {self._min_loss_tokens} trainable tokens"
            )
        payload = {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "image": record.get("image"),
            "prompt_text": record.get("prompt_text"),
        }
        return {"payload": payload}


def build_mm_dataset(
    dataset,
    tokenizer,
    processor,
    *,
    image_root: str,
    chat_template: Optional[str] = None,
    max_length: int = 2048,
    shuffle_seed: Optional[int] = 42,
    num_proc: Optional[int] = 8,
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
    train_only_last_turn: bool = False,
    minimum_valid_tokens: Optional[int] = None,
    loss_mask_filter: Optional[Callable[[object], bool]] = None,
    image_placeholder: str = "<|image_pad|>",
    image_token: str = "<|image_pad|>",
) -> Any:
    """Build the mixed text/image training dataset (mirrors build_eagle3_dataset).

    Input columns: ``conversations`` plus ``images`` (list of paths, empty =
    text-only). Output columns: ``input_ids``/``loss_mask``/``attention_mask``
    ([1, seq] tensors), ``image`` (str|None) and ``prompt_text`` (str|None).
    The processor runs once per unique image path (per map worker); span
    anchoring is text-tokenizer-only.
    """
    from .parse import GeneralParser
    from .preprocessing import preprocess_conversations

    if chat_template is None:
        raise ValueError("chat_template must be provided for the mixed dataset")
    assert chat_template in TEMPLATE_REGISTRY.get_all_template_names(), (
        f"Chat template {chat_template} not found in TEMPLATE_REGISTRY, "
        "you may need to register it first"
    )
    template: ChatTemplate = TEMPLATE_REGISTRY.get(chat_template)
    parser = GeneralParser(tokenizer, template)
    n_cache: dict = {}

    dataset = dataset.shuffle(seed=shuffle_seed)
    original_cols = dataset.column_names

    def preprocess_function(examples):
        conversations = examples["conversations"]
        images_col = examples.get("images")
        tools = examples.get("tools")
        n = len(conversations)
        results = {
            "input_ids": [None] * n,
            "loss_mask": [None] * n,
            "attention_mask": [None] * n,
            "image": [None] * n,
            "prompt_text": [None] * n,
        }
        text_indices = []
        for i in range(n):
            paths = images_col[i] if images_col else []
            if not paths:
                text_indices.append(i)
                continue
            input_ids, loss_mask, rendered = encode_image_conversation(
                parser,
                processor,
                paths,
                image_root,
                conversations[i],
                max_length,
                train_only_last_turn=train_only_last_turn,
                tool=(
                    tools[i]
                    if tools and isinstance(tools[i], list)
                    else None
                ),
                image_placeholder=image_placeholder,
                image_token=image_token,
                n_cache=n_cache,
            )
            results["input_ids"][i] = input_ids
            results["loss_mask"][i] = loss_mask
            results["attention_mask"][i] = torch.ones_like(loss_mask)
            results["image"][i] = paths[0]
            results["prompt_text"][i] = rendered

        if text_indices:
            tool_list = [
                tools[i] if tools and isinstance(tools[i], list) else []
                for i in text_indices
            ]
            processed = preprocess_conversations(
                tokenizer,
                [conversations[i] for i in text_indices],
                template,
                max_length,
                is_preformatted=False,
                train_only_last_turn=train_only_last_turn,
                tools=tool_list,
            )
            for slot, ids, mask in zip(
                text_indices, processed["input_ids"], processed["loss_mask"]
            ):
                results["input_ids"][slot] = ids
                results["loss_mask"][slot] = mask
        for slot in text_indices:
            results["attention_mask"][slot] = torch.ones_like(results["loss_mask"][slot])
        return results

    if cache_dir and cache_key:
        load_from_cache_file = True
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_name = os.path.join(cache_dir, f"{cache_key}.pkl")
        print(f"dataset is cached at {cache_file_name}")
    else:
        load_from_cache_file = False
        cache_file_name = None
        print("dataset is not cached")

    if num_proc is not None and num_proc > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        batch_size=1000,
        remove_columns=original_cols,
        load_from_cache_file=load_from_cache_file,
        cache_file_name=cache_file_name,
    )

    if minimum_valid_tokens is not None:
        before_filter = len(dataset)

        def has_minimum_valid_tokens(example):
            loss_mask = example["loss_mask"]
            if isinstance(loss_mask, torch.Tensor):
                valid_tokens = int(loss_mask.sum().item())
            else:
                valid_tokens = sum(
                    int(token)
                    for row in loss_mask
                    for token in (row if isinstance(row, list) else [row])
                )
            return valid_tokens >= minimum_valid_tokens

        dataset = dataset.filter(
            has_minimum_valid_tokens,
            num_proc=num_proc,
            desc=f"Filtering samples with >= {minimum_valid_tokens} trainable tokens",
        )
        print(
            f"Filtered dataset by trainable tokens: {before_filter} -> {len(dataset)}"
        )

    if loss_mask_filter is not None:
        before_filter = len(dataset)

        def has_eligible_loss_mask(example):
            return loss_mask_filter(example["loss_mask"])

        dataset = dataset.filter(
            has_eligible_loss_mask,
            num_proc=num_proc,
            desc="Filtering samples by algorithm loss-mask eligibility",
        )
        print(
            f"Filtered dataset by loss-mask eligibility: {before_filter} -> {len(dataset)}"
        )

    dataset.set_format(type="torch")
    return dataset
