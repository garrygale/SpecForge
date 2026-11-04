"""Image+text ``ServerInputAdapter`` for externally captured streaming training.

The draft stays text-only: multimodal samples only change what the producer
sends to the capture server (pre-expanded ``input_ids`` + base64 image data
URIs; the server injects vision embeddings at the image-token positions) and
the token sequence the trainer sees (the processor-expanded ``input_ids``,
where the image occupies a prompt-side token grid with ``loss_mask == 0``).

Modality identifier: ``"image"`` (register per algorithm alongside the text
provider; layout and collator are shared with text because the trainer-side
feature contract is unchanged).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import Any, Callable, Optional

MODALITY = "image"

_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
}


def load_processor(config):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        config.model.target_model_path,
        cache_dir=config.model.cache_dir,
        trust_remote_code=config.model.trust_remote_code,
    )


def _tokenizer_chat_template_hash(tokenizer) -> Optional[str]:
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return None
    return hashlib.sha256(str(template).encode("utf-8")).hexdigest()[:12]


def _processor_identity(processor) -> dict:
    return {
        "type": type(processor).__name__,
        "image_token": getattr(processor, "image_token", None),
    }


def mm_prompt_cache_key(config, *, tokenizer, processor) -> str:
    """Cache identity for the mixed dataset, independent of the text cache.

    Tokenization depends only on the data, tokenizer/template, and the image
    root/processor expansion, so draft weights/layer overrides deliberately
    do not participate (unlike the legacy text key, avoiding warm-start
    re-tokenization).
    """
    source_path = config.data.train_data_path or config.data.prompts_path
    content_hash = None
    if source_path and os.path.isfile(source_path):
        hasher = hashlib.sha256()
        with open(source_path, "rb") as source_file:
            for chunk in iter(lambda: source_file.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        content_hash = hasher.hexdigest()
    identity = {
        "path": source_path,
        "content_hash": content_hash,
        "max_length": config.data.max_length,
        "chat_template": config.data.chat_template,
        "tokenizer_chat_template_hash": _tokenizer_chat_template_hash(tokenizer),
        "train_only_last_turn": config.data.train_only_last_turn,
        "max_prompts": config.data.max_prompts,
        "target_model": config.model.target_model_path,
        "input_modality": MODALITY,
        "image_root": config.data.image_root,
        "processor": _processor_identity(processor),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class ImageTextAdapter:
    """``ServerInputAdapter`` implementation for mixed text/image JSONL.

    ``minimum_loss_tokens_fn`` / ``loss_mask_filter`` come from the owning
    algorithm's registration so dataset eligibility filtering matches the
    text path exactly while staying lazy inside the dataset pipeline.
    """

    def __init__(
        self,
        *,
        minimum_loss_tokens_fn: Optional[Callable[[Any, Any], int]] = None,
        loss_mask_filter: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        self._minimum_loss_tokens_fn = minimum_loss_tokens_fn
        self._loss_mask_filter = loss_mask_filter

    # -- ServerInputAdapter ---------------------------------------------------

    def load_input_tools(self, config):
        """Tokenizer only -- the consumer/trainer never needs the processor."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.target_model_path,
            cache_dir=config.model.cache_dir,
            trust_remote_code=config.model.trust_remote_code,
        )
        if config.model.tokenizer_pad_token_id is not None:
            tokenizer.pad_token_id = config.model.tokenizer_pad_token_id
        elif tokenizer.pad_token_id is None:
            fallback_id = tokenizer.eos_token_id
            if isinstance(fallback_id, (list, tuple)):
                fallback_id = fallback_id[0] if fallback_id else None
            if fallback_id is None:
                fallback_id = tokenizer.unk_token_id
            if fallback_id is None:
                raise ValueError(
                    "target tokenizer has no pad, EOS, or unknown token ID; set "
                    "model.tokenizer_pad_token_id explicitly"
                )
            tokenizer.pad_token_id = fallback_id
        return tokenizer

    def prepare_prompts(self, config, input_tools, *, draft_config) -> list:
        from datasets import load_dataset

        path = config.data.train_data_path or config.data.prompts_path
        if not path:
            raise ValueError("image modality requires data.train_data_path")
        if config.data.is_preformatted:
            raise ValueError(
                "image modality does not support pre-tokenized inputs; images "
                "must be expanded by the processor at map time"
            )
        processor = load_processor(config)
        tokenizer = input_tools
        image_token = getattr(processor, "image_token", None) or "<|image_pad|>"
        placeholder = image_token
        vision_start = getattr(processor, "vision_start_token", None) or getattr(
            tokenizer, "vision_start_token", None
        )
        vision_end = getattr(processor, "vision_end_token", None) or getattr(
            tokenizer, "vision_end_token", None
        )
        if vision_start and vision_end:
            placeholder = f"{vision_start}{image_token}{vision_end}"

        min_loss_tokens = 1
        if self._minimum_loss_tokens_fn is not None:
            min_loss_tokens = int(self._minimum_loss_tokens_fn(config, draft_config))

        cache_key = config.data.cache_key or mm_prompt_cache_key(
            config, tokenizer=tokenizer, processor=processor
        )

        from specforge.data.mm_preprocessing import MMPromptSequence, build_mm_dataset

        dataset = load_dataset("json", data_files=path, split="train")
        limit = (
            None
            if config.data.max_prompts in (None, 0)
            else int(config.data.max_prompts)
        )
        if (
            self._loss_mask_filter is None
            and limit is not None
            and limit < len(dataset)
        ):
            dataset = dataset.select(range(limit))

        processed = build_mm_dataset(
            dataset,
            tokenizer,
            processor,
            image_root=config.data.image_root,
            chat_template=config.data.chat_template,
            max_length=config.data.max_length,
            num_proc=config.data.build_dataset_num_proc,
            cache_dir=config.data.cache_dir,
            cache_key=cache_key,
            train_only_last_turn=config.data.train_only_last_turn,
            minimum_valid_tokens=min_loss_tokens,
            loss_mask_filter=self._loss_mask_filter,
            image_placeholder=placeholder,
            image_token=image_token,
        )
        return MMPromptSequence(
            processed,
            max_length=config.data.max_length,
            min_loss_tokens=min_loss_tokens,
            loss_mask_filter=self._loss_mask_filter,
        )

    def split_request_groups(self, tasks):
        """All tasks share one input mode (input_ids), so a mixed batch is a
        single server call; kept for interface parity/testing."""
        return [list(tasks)]

    def build_request_inputs(self, tasks):
        """One request body for any mix of text and image tasks.

        Every task travels as exact-token ``input_ids``. Image tasks carry
        the processor-EXPANDED ids (image grid tokens included) plus their
        image as a base64 data URI; the server injects vision embeddings at
        the image-token positions -- the standard pre-tokenized multimodal
        request (verl-style), which also works for engines launched with
        ``--skip-tokenizer-init`` that reject text prompts.
        """
        payload_images = [task.payload.get("image") for task in tasks]
        input_ids = [list(task.payload["input_ids"]) for task in tasks]
        if not any(payload_images):
            return {"input_ids": input_ids}
        image_data = []
        for task, rel in zip(tasks, payload_images):
            if not rel:
                # per-request None entries are normalized upstream
                image_data.append(None)
                continue
            path = rel if os.path.isabs(rel) else os.path.join(self._image_root(), rel)
            with open(path, "rb") as fh:
                data = fh.read()
            ext = os.path.splitext(rel)[1].lstrip(".").lower()
            mime = _MIME.get(ext, "image/jpeg")
            image_data.append(
                [f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"]
            )
        return {"input_ids": input_ids, "image_data": image_data}

    def _image_root(self) -> str:
        config = getattr(self, "_config", None)
        if config is None:
            raise RuntimeError(
                "adapter was not built through image_streaming_adapter(); "
                "image_root is unknown"
            )
        return config.data.image_root

    # -- config capture ---------------------------------------------------------

    def bind(self, config) -> "ImageTextAdapter":
        self._config = config
        return self


def image_streaming_adapter(
    *,
    minimum_loss_tokens_fn: Optional[Callable[[Any, Any], int]] = None,
    loss_mask_filter: Optional[Callable[[Any], bool]] = None,
) -> Callable[[Any], ImageTextAdapter]:
    """Provider factory: build the adapter, capturing algorithm eligibility."""

    def build(config):
        return ImageTextAdapter(
            minimum_loss_tokens_fn=minimum_loss_tokens_fn,
            loss_mask_filter=loss_mask_filter,
        ).bind(config)

    return build


__all__ = [
    "MODALITY",
    "ImageTextAdapter",
    "image_streaming_adapter",
    "load_processor",
    "mm_prompt_cache_key",
]
