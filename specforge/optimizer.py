import copy
import logging
import math

import torch
import torch.distributed as dist

from specforge.lr_scheduler import ConstantWarmupLR, CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0

logger = logging.getLogger(__name__)


#: Name substrings that force a param onto the AdamW fallback group even when
#: it is a full 2D matrix. Embeddings and vocab projections have row-sparse
#: gradients (only the rows touched by the batch receive updates), so Newton-
#: Schulz orthogonalization would mix unrelated rows; the table-like params
#: (flare fusion weights, HCA position bias) have the same per-row structure.
#: This is the exact failure the speculators Muon port hit with the DSpark
#: markov head (fixed upstream in vllm-project/speculators#971).
_MUON_ADAMW_NAME_HINTS = (
    "embed_tokens",
    "lm_head",
    "markov_w1",
    "markov_w2",
    "codebook",
    "layer_fusion_weights",
    "position_bias",
)


def split_params_for_muon(named_params, vocab_dims=(), name_hints=()):
    """Split trainable ``named_parameters()`` into (muon, adamw) lists.

    Muon gets full 2D hidden matrices with both dims > 1, no hint in the name
    and no vocab-sized dimension. Everything else — norms/biases, embeddings,
    vocab projections, table-like params and 3D tensors — stays on AdamW.
    """
    vocab_dims = {int(dim) for dim in vocab_dims if dim}
    hints = tuple(_MUON_ADAMW_NAME_HINTS) + tuple(name_hints)
    muon, adamw = [], []
    for name, param in named_params:
        if (
            param.ndim == 2
            and min(param.shape) > 1
            and max(param.shape) not in vocab_dims
            and not any(hint in name for hint in hints)
        ):
            muon.append((name, param))
        else:
            adamw.append((name, param))
    return muon, adamw


class BF16Optimizer:
    """AdamW — optionally Muon for the 2D hidden matrices — over fp32 master
    copies of the bf16 trainable params, with grad clipping and configurable
    warmup scheduling."""

    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        lr_scheduler="cosine",
        offload_master=False,
        optimizer_type="adamw",
        muon_lr=None,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_weight_decay=0.0,
        muon_ns_steps=5,
        muon_adjust_lr="match_rms_adamw",
        muon_adamw_name_hints=(),
    ):
        # defaults copied from EAGLE traineagle3 ds_config.json
        self.model = model
        self._named_model_params = [
            (name, p) for name, p in model.named_parameters() if p.requires_grad
        ]
        self.model_params = [p for _, p in self._named_model_params]
        if optimizer_type not in ("adamw", "muon"):
            raise ValueError(
                f"unsupported optimizer={optimizer_type!r}; "
                "expected 'adamw' or 'muon'"
            )
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.muon_optimizer = None
        self.max_grad_norm = max_grad_norm
        self.offload_master = bool(offload_master)
        self.fp32_params = [
            (
                p.detach().to(device="cpu", dtype=torch.float32).clone()
                if self.offload_master
                else p.detach().clone().to(torch.float32)
            )
            for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        adamw_masters = self.fp32_params
        if optimizer_type == "muon":
            adamw_masters = self._build_muon_optimizer(
                lr=lr,
                muon_lr=muon_lr,
                muon_momentum=muon_momentum,
                muon_nesterov=muon_nesterov,
                muon_weight_decay=muon_weight_decay,
                muon_ns_steps=muon_ns_steps,
                muon_adjust_lr=muon_adjust_lr,
                muon_adamw_name_hints=muon_adamw_name_hints,
            )
        self.optimizer = torch.optim.AdamW(
            adamw_masters, lr=lr, weight_decay=weight_decay
        )
        self.last_grad_norm = None
        self._grad_norm_process_group = None
        self._reduce_grad_norm_across_ranks = True
        scheduler_types = {
            "constant": ConstantWarmupLR,
            "cosine": CosineAnnealingWarmupLR,
        }
        if lr_scheduler not in scheduler_types:
            raise ValueError(
                f"unsupported lr_scheduler={lr_scheduler!r}; "
                f"expected one of {sorted(scheduler_types)}"
            )
        self.lr_scheduler_type = lr_scheduler
        self.scheduler = scheduler_types[lr_scheduler](
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )
        if self.muon_optimizer is not None:
            # The scheduler init already moved the AdamW group lr to the first
            # warmup value; mirror it so step 1 uses the same warmup scale.
            self._sync_muon_lr()

    def _build_muon_optimizer(
        self,
        *,
        lr,
        muon_lr,
        muon_momentum,
        muon_nesterov,
        muon_weight_decay,
        muon_ns_steps,
        muon_adjust_lr,
        muon_adamw_name_hints,
    ):
        """Create the Muon inner optimizer over the 2D-matrix master subset
        and return the remaining masters for the AdamW fallback."""
        try:
            from torch.optim import Muon
        except ImportError as exc:
            raise ImportError(
                "optimizer='muon' needs torch>=2.9 (torch.optim.Muon); "
                f"this environment has torch {torch.__version__}"
            ) from exc
        vocab_size = getattr(
            getattr(self.model, "config", None), "vocab_size", None
        )
        muon_named, adamw_named = split_params_for_muon(
            self._named_model_params,
            vocab_dims=(vocab_size,) if vocab_size else (),
            name_hints=muon_adamw_name_hints,
        )
        muon_ids = {id(p) for _, p in muon_named}
        muon_masters = [
            mp
            for mp, (_, p) in zip(self.fp32_params, self._named_model_params)
            if id(p) in muon_ids
        ]
        adamw_masters = [
            mp
            for mp, (_, p) in zip(self.fp32_params, self._named_model_params)
            if id(p) not in muon_ids
        ]
        if not adamw_masters:
            raise ValueError(
                "optimizer='muon' found no param for the AdamW fallback "
                "group; every real draft carries 1D norm/bias params, so "
                "this model cannot be trained in muon mode"
            )
        self._muon_lr_base = lr if muon_lr is None else muon_lr
        if muon_masters:
            self.muon_optimizer = Muon(
                muon_masters,
                lr=self._muon_lr_base,
                momentum=muon_momentum,
                nesterov=muon_nesterov,
                weight_decay=muon_weight_decay,
                ns_steps=muon_ns_steps,
                adjust_lr_fn=muon_adjust_lr,
            )
        else:
            logger.warning(
                "optimizer='muon' but no trainable param matched the Muon "
                "group; every param stays on AdamW"
            )
        muon_numel = sum(p.numel() for _, p in muon_named)
        adamw_numel = sum(p.numel() for _, p in adamw_named)
        listing = ["Muon param split:"]
        listing += [f"  muon  {name} {tuple(p.shape)}" for name, p in muon_named]
        listing += [f"  adamw {name} {tuple(p.shape)}" for name, p in adamw_named]
        listing.append(
            f"  total: muon {len(muon_named)} params / "
            f"{muon_numel / 1e6:.2f}M elements, adamw {len(adamw_named)} "
            f"params / {adamw_numel / 1e6:.2f}M elements"
        )
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        ):
            logger.info("\n".join(listing))
        return adamw_masters

    def _sync_muon_lr(self):
        """Mirror the scheduled AdamW lr onto the Muon group, preserving the
        configured muon_lr/learning_rate ratio across warmup and decay."""
        ratio = self._muon_lr_base / self.lr
        current = self.optimizer.param_groups[0]["lr"]
        for group in self.muon_optimizer.param_groups:
            group["lr"] = current * ratio

    def configure_grad_norm_reduction(
        self, *, process_group=None, enabled: bool = True
    ) -> None:
        """Configure the group that owns disjoint gradient shards.

        FSDP backends disable the reduction for replicated/NO_SHARD parameters.
        """
        self._grad_norm_process_group = process_group
        self._reduce_grad_norm_across_ranks = enabled

    def _reduce_grad_norm(self, total_norm_sq):
        """All-reduce the squared L2 norm across shard ranks and derive the
        clip coefficient.

        ``total_norm_sq`` must already live on a device the process group can
        reduce (e.g. CUDA for NCCL). Returns ``(total_norm, clip_coef)``.
        """
        if (
            self._reduce_grad_norm_across_ranks
            and dist.is_available()
            and dist.is_initialized()
        ):
            dist.all_reduce(
                total_norm_sq,
                op=dist.ReduceOp.SUM,
                group=self._grad_norm_process_group,
            )
        total_norm = total_norm_sq.sqrt()
        clip_coef = torch.clamp(self.max_grad_norm / (total_norm + 1e-6), max=1.0)
        return total_norm, clip_coef

    def _grad_norm_and_clip_coefficient(self):
        """Compute the global grad norm from the model params on their own
        device, where NCCL can reduce it safely, without materialising master
        gradients first."""
        grads = [p.grad.detach() for p in self.model_params if p.grad is not None]
        if grads:
            total_norm_sq = torch.stack(
                [grad.float().square().sum() for grad in grads]
            ).sum()
        else:
            device = self.model_params[0].device if self.model_params else "cpu"
            total_norm_sq = torch.zeros((), dtype=torch.float32, device=device)
        return self._reduce_grad_norm(total_norm_sq)

    def _clip_grad_norm(self):
        """Clip already-populated FP32 master gradients in place.

        Convenience entry point for optimizer tests and custom loops. When
        masters are CPU-offloaded, only the scalar norm is moved to the model
        device so a NCCL process group can still participate in the reduction.
        """
        grads = [master.grad for master in self.fp32_params if master.grad is not None]
        if grads:
            local_norm_sq = torch.stack(
                [grad.float().square().sum() for grad in grads]
            ).sum()
        else:
            master_device = self.fp32_params[0].device if self.fp32_params else "cpu"
            local_norm_sq = torch.zeros((), dtype=torch.float32, device=master_device)

        reduction_device = (
            self.model_params[0].device if self.model_params else local_norm_sq.device
        )
        total_norm, clip_coef = self._reduce_grad_norm(
            local_norm_sq.to(reduction_device)
        )
        if not math.isfinite(float(total_norm.item())):
            return total_norm
        for grad in grads:
            coefficient = (
                clip_coef
                if clip_coef.device == grad.device
                else float(clip_coef.item())
            )
            grad.mul_(coefficient)
        return total_norm

    def step(self):
        grad_norm, clip_coefficient = self._grad_norm_and_clip_coefficient()
        norm_value = float(grad_norm.item())
        if not math.isfinite(norm_value):
            # One inf/nan grad makes the clip coefficient 0, and 0 * inf turns
            # every fp32 master permanently NaN. Drop the step instead; grads
            # must still be cleared or the poison accumulates into the next
            # backward and the guard would fire forever. The norm is already
            # reduced across ranks (or replicated by DDP under NO_SHARD), so
            # every rank takes this branch together, and the branch itself
            # issues no collectives.
            logger.warning(
                "BF16Optimizer: non-finite grad norm (%s); skipping step", norm_value
            )
            with torch.no_grad():
                for p in self.model_params:
                    p.grad = None
            self.last_grad_norm = grad_norm.detach()
            return self.last_grad_norm
        cpu_clip_coefficient = (
            float(clip_coefficient.item()) if self.offload_master else None
        )
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                if p.grad is None:
                    mp.grad = None
                    continue
                master_grad = p.grad.detach().to(
                    device=mp.device,
                    dtype=torch.float32,
                )
                master_grad.mul_(
                    cpu_clip_coefficient
                    if cpu_clip_coefficient is not None
                    else clip_coefficient
                )
                mp.grad = master_grad
        self.last_grad_norm = grad_norm.detach()
        self.optimizer.step()
        if self.muon_optimizer is not None:
            self.muon_optimizer.step()
        self.optimizer.zero_grad()
        if self.muon_optimizer is not None:
            self.muon_optimizer.zero_grad()
        self.scheduler.step()
        if self.muon_optimizer is not None:
            self._sync_muon_lr()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(device=p.device, dtype=p.dtype))
                p.grad = None
        return self.last_grad_norm

    def load_state_dict(self, state_dict):
        """Restore optimizer/scheduler state and, when present, the rank-local
        fp32 master params; without them the masters are re-cloned from the
        bf16 weights and the resume is not numerically faithful."""
        saved_scheduler_type = state_dict.get("lr_scheduler_type", "cosine")
        if saved_scheduler_type != self.lr_scheduler_type:
            raise ValueError(
                "checkpoint optimizer used lr_scheduler="
                f"{saved_scheduler_type!r} but this run has "
                f"lr_scheduler={self.lr_scheduler_type!r}"
            )
        saved_max_grad_norm = state_dict.get("max_grad_norm")
        if saved_max_grad_norm is not None and float(saved_max_grad_norm) != float(
            self.max_grad_norm
        ):
            raise ValueError(
                "checkpoint optimizer used max_grad_norm="
                f"{saved_max_grad_norm} but this run has "
                f"max_grad_norm={self.max_grad_norm}"
            )
        saved_optimizer_type = state_dict.get("optimizer_type", "adamw")
        if saved_optimizer_type != self.optimizer_type:
            raise ValueError(
                "checkpoint optimizer was "
                f"{saved_optimizer_type!r} but this run uses "
                f"optimizer={self.optimizer_type!r}; momentum semantics "
                "differ between the two, so resuming across optimizer types "
                "is not supported"
            )
        # offload_master is a pure device-placement choice: restored fp32
        # masters and Adam moments are relocated to the current master device,
        # so toggling it on resume is safe and intentionally not gated here.
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        print_on_rank0("Successfully loaded optimizer state_dict.")
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded scheduler state_dict.")
        if self.optimizer_type == "muon":
            muon_state = state_dict.get("muon_state_dict")
            if self.muon_optimizer is None:
                if muon_state is not None:
                    raise ValueError(
                        "checkpoint carries muon_state_dict but no trainable "
                        "param of this model matched the Muon group"
                    )
                logger.warning(
                    "optimizer='muon' with an empty Muon group: nothing to "
                    "restore beyond the AdamW/scheduler state"
                )
            elif muon_state is None:
                raise ValueError(
                    "checkpoint carries optimizer_type='muon' but no "
                    "muon_state_dict; the payload is incomplete"
                )
            else:
                self.muon_optimizer.load_state_dict(muon_state)
                print_on_rank0("Successfully loaded Muon state_dict.")
                self._sync_muon_lr()
        saved_fp32 = state_dict.get("fp32_params")
        if saved_fp32 is not None:
            if len(saved_fp32) != len(self.fp32_params):
                raise ValueError(
                    f"checkpoint carries {len(saved_fp32)} fp32 master params "
                    f"but this rank has {len(self.fp32_params)}"
                )
            with torch.no_grad():
                for i, (saved, mp) in enumerate(zip(saved_fp32, self.fp32_params)):
                    if saved.shape != mp.shape:
                        raise ValueError(
                            f"fp32 master param {i} shape mismatch: checkpoint "
                            f"{tuple(saved.shape)} vs current {tuple(mp.shape)}"
                        )
                    mp.data.copy_(saved.to(mp.device, mp.dtype))
        else:
            logger.warning(
                "checkpoint has no fp32_params; re-cloning master params from "
                "bf16 weights — resume will not be numerically faithful"
            )
            with torch.no_grad():
                for p, mp in zip(self.model_params, self.fp32_params):
                    mp.data.copy_(p.detach().to(device=mp.device, dtype=mp.dtype))

    def state_dict(self):
        # torch hands back live references from optimizer.state_dict(); clone
        # everything so a kept payload cannot be mutated by further steps or
        # alias another optimizer loaded from it in-process.
        state = {
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler_state_dict": copy.deepcopy(self.scheduler.state_dict()),
            "lr_scheduler_type": self.lr_scheduler_type,
            "optimizer_type": self.optimizer_type,
            "max_grad_norm": self.max_grad_norm,
            # rank-local fp32 masters; without them a resume re-quantizes from bf16
            "fp32_params": [t.detach().to("cpu", copy=True) for t in self.fp32_params],
        }
        if self.muon_optimizer is not None:
            state["muon_state_dict"] = copy.deepcopy(
                self.muon_optimizer.state_dict()
            )
        return state

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
