# coding=utf-8
"""Anchor-mining regressions: chain walk, override plumbing, family variants.

The drafts below are scripted: they decode (anchor, slot) from the draft
position ids and replay a fixed token per slot, so the mined chain is
assertable by hand against an independent reference walk.
"""

import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]

_stub_dflash_draft = types.ModuleType("specforge.modeling.draft.dflash")


class _DFlashDraftStub(nn.Module):
    pass


_stub_dflash_draft.DFlashDraftModel = _DFlashDraftStub
_stub_dflash_draft.resolve_sliding_draft_causal = lambda config: True

_spec = importlib.util.spec_from_file_location(
    "specforge.algorithms.common.dflash_family_model",
    REPO / "specforge" / "algorithms" / "common" / "dflash_family_model.py",
)
_dflash_module = importlib.util.module_from_spec(_spec)

_pkg_specforge = types.ModuleType("specforge")
_pkg_specforge.__path__ = [str(REPO / "specforge")]
_pkg_algorithms = types.ModuleType("specforge.algorithms")
_pkg_algorithms.__path__ = [str(REPO / "specforge" / "algorithms")]
_pkg_common = types.ModuleType("specforge.algorithms.common")
_pkg_common.__path__ = [str(REPO / "specforge" / "algorithms" / "common")]
_pkg_modeling = types.ModuleType("specforge.modeling")
_pkg_modeling.__path__ = [str(REPO / "specforge" / "modeling")]
_pkg_draft = types.ModuleType("specforge.modeling.draft")
_pkg_draft.__path__ = [str(REPO / "specforge" / "modeling" / "draft")]

with patch.dict(
    sys.modules,
    {
        "specforge": _pkg_specforge,
        "specforge.algorithms": _pkg_algorithms,
        "specforge.algorithms.common": _pkg_common,
        "specforge.algorithms.common.dflash_family_model": _dflash_module,
        "specforge.modeling": _pkg_modeling,
        "specforge.modeling.draft": _pkg_draft,
        "specforge.modeling.draft.dflash": _stub_dflash_draft,
    },
):
    _spec.loader.exec_module(_dflash_module)
OnlineDFlashModel = _dflash_module.OnlineDFlashModel
OnlineDominoModel = _dflash_module.OnlineDominoModel
walk_anchor_chain = _dflash_module.walk_anchor_chain


class _ScriptedDraft(nn.Module):
    """Emits one-hot(script token) per (anchor, slot), scaled by ``gain``."""

    def __init__(self, vocab_size, block_size, seq_len, script, default_token=0):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.script = script
        self.default_token = default_token
        self.block_size = block_size
        self.seq_len = seq_len
        self.sliding_window = None
        self.vocab_size = vocab_size

    def forward(self, position_ids, noise_embedding, target_hidden,
                attention_mask=None, **kwargs):
        del noise_embedding, target_hidden, attention_mask, kwargs
        draft_positions = position_ids[:, self.seq_len:]
        n_rows = draft_positions.shape[1]
        onehot = torch.zeros(n_rows, self.vocab_size)
        for r in range(n_rows):
            slot = r % self.block_size
            anchor = int(draft_positions[0, r]) - slot
            onehot[r, self.script.get((anchor, slot), self.default_token)] = 1.0
        return onehot.unsqueeze(0) * self.gain


class _ScriptedDominoDraft(_ScriptedDraft):
    """Slot-0 prefix keeps the base token; suffix slots add a scripted spike."""

    def __init__(self, *args, base_token=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.suffix_start = 1
        self.shift_label = True
        self.base_token = base_token
        self._anchors = []

    def forward(self, position_ids, noise_embedding, target_hidden,
                attention_mask=None, **kwargs):
        draft_positions = position_ids[:, self.seq_len:]
        n_rows = draft_positions.shape[1]
        self._anchors = []
        onehot = torch.zeros(n_rows, self.vocab_size)
        for r in range(n_rows):
            slot = r % self.block_size
            anchor = int(draft_positions[0, r]) - slot
            if slot == 0:
                self._anchors.append(anchor)
            onehot[r, self.base_token] = 1.0
        return onehot.unsqueeze(0) * self.gain

    def apply_logits_head(self, base_logits, *, prev_token_ids=None,
                          prev_token_embeddings=None, hidden_states=None,
                          lm_head=None):
        del prev_token_ids, prev_token_embeddings, hidden_states, lm_head
        out = base_logits.clone()
        n_blocks, block_size = out.shape[1], out.shape[2]
        for i in range(n_blocks):
            anchor = self._anchors[i]
            for slot in range(self.suffix_start, block_size):
                out[0, i, slot, self.script.get(
                    (anchor, slot), self.default_token)] += 5.0
        return out


def _eye_head(vocab_size):
    head = nn.Linear(vocab_size, vocab_size, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(vocab_size))
    return head


def _manual_chain(script, ids, mask, start, block, shift, default_token=0,
                  prefix_token=None):
    seq_len = ids.shape[1]
    ids_list = ids[0].tolist()
    mask_list = (mask[0] > 0.5).tolist()
    chain = []
    a = start
    while a + 1 < seq_len and mask_list[a + 1]:
        chain.append(a)
        stop = None
        for k in range(1 - shift, block):
            pos = a + k + shift
            if pos >= seq_len or not mask_list[pos]:
                stop = k
                break
            if prefix_token is not None and k == 0:
                pred = prefix_token
            else:
                pred = script.get((a, k), default_token)
            if pred != ids_list[pos]:
                stop = k
                break
        a = a + (block if stop is None else stop) + shift
    return chain


class TestWalkAnchorChain(unittest.TestCase):
    def test_base_axis_rejection_then_full_accept(self):
        stop = torch.zeros(3, 4, dtype=torch.bool)
        stop[0, 2] = True
        anchorable = torch.ones(3, dtype=torch.bool)
        chain, full = walk_anchor_chain(stop, anchorable, 0, 0, 0)
        self.assertEqual(chain, [0, 2])
        self.assertEqual(full, 1)

    def test_shifted_axis_slot_zero_is_a_proposal(self):
        stop = torch.zeros(4, 4, dtype=torch.bool)
        stop[0, 0] = True
        stop[1, 1] = True
        anchorable = torch.ones(4, dtype=torch.bool)
        chain, full = walk_anchor_chain(stop, anchorable, 0, 0, 1)
        self.assertEqual(chain, [0, 1, 3])
        self.assertEqual(full, 1)

    def test_offset_and_drain(self):
        stop = torch.zeros(5, 4, dtype=torch.bool)
        anchorable = torch.zeros(5, dtype=torch.bool)
        anchorable[1] = True
        chain, full = walk_anchor_chain(stop, anchorable, 6, 5, 0)
        self.assertEqual(chain, [6])
        self.assertEqual(full, 1)


class TestBaseMining(unittest.TestCase):
    vocab = 9
    block = 4
    seq_len = 20
    prompt = 5

    def _ids(self):
        torch.manual_seed(0)
        ids = torch.randint(1, self.vocab, (1, self.seq_len))
        torch.manual_seed(1)
        hidden = torch.randn(1, self.seq_len, 8)
        return ids, hidden

    def _setup(self, ids, hidden, script=None, **model_kwargs):
        mask = torch.zeros(1, self.seq_len)
        mask[0, self.prompt :] = 1.0
        draft = _ScriptedDraft(self.vocab, self.block, self.seq_len, script or {})
        model_kwargs.setdefault("anchor_policy", "mined")
        model_kwargs.setdefault("anchor_mining_chunk_candidates", 512)
        return OnlineDFlashModel(
            draft_model=draft,
            target_lm_head=_eye_head(self.vocab),
            target_embed_tokens=nn.Embedding(self.vocab, self.vocab),
            mask_token_id=0,
            block_size=self.block,
            attention_backend="sdpa",
            **model_kwargs,
        ), mask

    def _kept(self, model, ids, hidden, mask):
        anchors, keep = model._mine_anchor_positions(ids, hidden, mask)
        return anchors, keep, anchors[0][keep[0]].tolist()

    def test_matches_manual_walk_all_mismatch(self):
        ids, hidden = self._ids()
        model, mask = self._setup(ids, hidden, num_anchors=64)
        _, _, kept = self._kept(model, ids, hidden, mask)
        expected = _manual_chain({}, ids, mask, self.prompt - 1, self.block, 0)
        self.assertEqual(kept, expected)
        self.assertEqual(len(expected), self.seq_len - self.prompt)

    def test_matches_manual_walk_with_jumps(self):
        ids, hidden = self._ids()
        script = {
            (4, 1): int(ids[0, 5]),
            (4, 2): int(ids[0, 6]),
            (7, 1): int(ids[0, 8]),
            (7, 2): int(ids[0, 9]),
            (7, 3): int(ids[0, 10]),
        }
        model, mask = self._setup(ids, hidden, script=script, num_anchors=64)
        _, _, kept = self._kept(model, ids, hidden, mask)
        expected = _manual_chain(
            script, ids, mask, self.prompt - 1, self.block, 0
        )
        self.assertEqual(kept, expected)
        self.assertEqual(expected[:3], [4, 7, 11])

    def test_chunk_size_invariance(self):
        ids, hidden = self._ids()
        model, mask = self._setup(ids, hidden, num_anchors=64)
        a1, k1, _ = self._kept(model, ids, hidden, mask)
        model.anchor_mining_chunk_candidates = 3
        a2, k2, _ = self._kept(model, ids, hidden, mask)
        self.assertTrue(torch.equal(a1, a2))
        self.assertTrue(torch.equal(k1, k2))

    def test_subsample_even_and_deterministic(self):
        ids, hidden = self._ids()
        model, mask = self._setup(ids, hidden, num_anchors=5)
        anchors, keep, kept = self._kept(model, ids, hidden, mask)
        expected = _manual_chain({}, ids, mask, self.prompt - 1, self.block, 0)
        self.assertEqual(len(kept), 5)
        self.assertEqual(kept[0], expected[0])
        self.assertEqual(kept[-1], expected[-1])
        self.assertEqual(kept, sorted(kept))
        self.assertTrue(all(k in expected for k in kept))
        self.assertTrue(bool(keep[0].all()))

    def test_sentinel_padding_when_chain_short(self):
        ids, hidden = self._ids()
        model, mask = self._setup(ids, hidden, num_anchors=64)
        anchors, keep, kept = self._kept(model, ids, hidden, mask)
        expected = _manual_chain({}, ids, mask, self.prompt - 1, self.block, 0)
        self.assertLess(len(expected), 64)
        self.assertEqual(kept, expected)
        self.assertTrue(bool((anchors[0][len(expected) :] == 0).all()))
        self.assertTrue(bool((~keep[0][len(expected) :]).all()))

    def test_sentinels_do_not_change_loss(self):
        ids, hidden = self._ids()
        model, mask = self._setup(ids, hidden, num_anchors=64)
        loss_wide, _, _ = model(ids, hidden, mask)
        _, keep = model._mine_anchor_positions(ids, hidden, mask)
        count = int(keep[0].sum())
        narrow = copy.deepcopy(model)
        narrow.num_anchors = count
        loss_narrow, _, _ = narrow(ids, hidden, mask)
        # zero-weight pads only change the reduction tree, not the value
        self.assertTrue(
            torch.allclose(loss_wide, loss_narrow, rtol=1e-6, atol=1e-7)
        )

    def test_mined_forward_equals_sampler_at_same_anchors(self):
        ids, hidden = self._ids()
        model, mask = self._setup(ids, hidden, num_anchors=6)
        anchors, keep = model._mine_anchor_positions(ids, hidden, mask)
        loss_mined, acc_mined, _ = model(ids, hidden, mask)
        model.anchor_policy = "random"
        model._sample_anchor_positions = MethodType(
            lambda self, *a, **k: (anchors, keep), model
        )
        loss_sampled, acc_sampled, _ = model(ids, hidden, mask)
        self.assertTrue(torch.equal(loss_mined, loss_sampled))
        self.assertTrue(torch.equal(acc_mined, acc_sampled))

    def test_mined_stats_reported(self):
        ids, hidden = self._ids()
        model, mask = self._setup(ids, hidden, num_anchors=64)
        _, _, metrics = model(ids, hidden, mask)
        expected = _manual_chain({}, ids, mask, self.prompt - 1, self.block, 0)
        count = metrics["ratio_metrics"]["mined_anchor_count"]
        self.assertAlmostEqual(
            float(count[0] / count[1]), float(len(expected)), places=5
        )

    def test_lambda_and_tv_objectives_run_with_mined_anchors(self):
        for lk in ("lambda", "tv"):
            with self.subTest(lk=lk):
                ids, hidden = self._ids()
                model, mask = self._setup(
                    ids, hidden, num_anchors=6, lk_loss_type=lk,
                    loss_decay_gamma=7.0,
                )
                loss, _, _ = model(ids, hidden, mask)
                self.assertTrue(torch.isfinite(loss).all())
                loss.backward()
                self.assertIsNotNone(model.draft_model.gain.grad)

    def test_all_masked_input_raises(self):
        ids, hidden = self._ids()
        model, _ = self._setup(ids, hidden, num_anchors=6)
        mask = torch.zeros(1, self.seq_len)
        with self.assertRaises(ValueError):
            model._mine_anchor_positions(ids, hidden, mask)

    def test_invalid_policy_rejected(self):
        ids, hidden = self._ids()
        with self.assertRaises(ValueError):
            self._setup(ids, hidden, num_anchors=6, anchor_policy="bogus")


class TestDominoMining(unittest.TestCase):
    vocab = 9
    block = 4
    seq_len = 20
    prompt = 5

    def _model(self, script, base_token=3, **model_kwargs):
        torch.manual_seed(2)
        draft = _ScriptedDominoDraft(
            self.vocab, self.block, self.seq_len, script, base_token=base_token
        )
        model_kwargs.setdefault("shift_label", True)
        model_kwargs.setdefault("anchor_policy", "mined")
        model_kwargs.setdefault("anchor_mining_chunk_candidates", 512)
        model = OnlineDominoModel(
            draft_model=draft,
            target_lm_head=_eye_head(self.vocab),
            target_embed_tokens=nn.Embedding(self.vocab, self.vocab),
            mask_token_id=0,
            block_size=self.block,
            attention_backend="sdpa",
            **model_kwargs,
        )
        return model

    def test_shifted_axis_matches_manual_walk(self):
        torch.manual_seed(3)
        ids = torch.randint(1, self.vocab, (1, self.seq_len))
        ids[0, 5] = 3
        mask = torch.zeros(1, self.seq_len)
        mask[0, self.prompt :] = 1.0
        hidden = torch.randn(1, self.seq_len, 8)
        script = {
            (4, 1): int(ids[0, 6]),
            (4, 2): int(ids[0, 7]),
            (4, 3): int(ids[0, 8]),
            (10, 1): int(ids[0, 11]),
        }
        model = self._model(script)
        anchors, keep = model._mine_anchor_positions(ids, hidden, mask)
        kept = anchors[0][keep[0]].tolist()
        expected = _manual_chain(
            script, ids, mask, self.prompt - 1, self.block, 1, prefix_token=3
        )
        self.assertEqual(kept, expected)
        self.assertEqual(expected[0], 4)
        self.assertIn(9, kept)

    def test_domino_full_accept_advances_block_plus_one(self):
        ids = torch.full((1, self.seq_len), 3, dtype=torch.long)
        mask = torch.zeros(1, self.seq_len)
        mask[0, self.prompt :] = 1.0
        hidden = torch.randn(1, self.seq_len, 8)
        script = {
            (4, 1): 3,
            (4, 2): 3,
            (4, 3): 3,
            (9, 1): 3,
            (9, 2): 3,
            (9, 3): 3,
            (14, 1): 3,
            (14, 2): 3,
            (14, 3): 3,
        }
        model = self._model(script)
        anchors, keep = model._mine_anchor_positions(ids, hidden, mask)
        kept = anchors[0][keep[0]].tolist()
        self.assertEqual(kept, [4, 9, 14])


if __name__ == "__main__":
    unittest.main()
