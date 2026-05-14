"""End-to-end runner: prefill -> compress KV cache -> decode/score with the
compressed cache. Handles the position_ids bookkeeping that becomes necessary
once tokens are dropped from the middle of the cache.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .compress import apply_compression, count_memory_cost, memory_ratio
from .importance import compute_importance


def _to_legacy(past) -> tuple:
    """Normalize HF past_key_values to a legacy tuple-of-tuples format."""
    if past is None:
        return ()
    if isinstance(past, tuple):
        return past
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    raise TypeError(f"Unsupported past_key_values type: {type(past)}")


def _to_cache(past_tuple: tuple) -> DynamicCache:
    """Wrap a legacy tuple cache as a DynamicCache for HF >= 4.50."""
    return DynamicCache.from_legacy_cache(past_tuple)


PolicyFn = Callable[[torch.Tensor, int], list[str]]


@dataclass
class CompressionConfig:
    """Hyper-parameters for the importance score and BM-KV allocation."""

    alpha: float = 0.5  # weight on attention contribution
    beta: float = 0.3   # weight on recency
    gamma: float = 0.2  # weight on prefix bonus
    lambd: float = 0.01  # decay rate for recency
    prefix_len: int = 4  # treat first N tokens as prefix anchors
    last_n_layers: int | None = None  # how many top layers to aggregate

    # Score variants used by some baselines
    use_attention: bool = True
    use_recency: bool = True
    use_prefix: bool = True

    def importance_kwargs(self) -> dict:
        return dict(
            alpha=self.alpha if self.use_attention else 0.0,
            beta=self.beta if self.use_recency else 0.0,
            gamma=self.gamma if self.use_prefix else 0.0,
            lambd=self.lambd,
            prefix_len=self.prefix_len,
            last_n_layers=self.last_n_layers,
        )


@dataclass
class CompressionStats:
    seq_len: int
    n_fp16: int
    n_int8: int
    n_drop: int
    cost: int
    memory_ratio: float
    actions: list[str] = field(repr=False)


class CompressedRunner:
    """Wraps a HuggingFace causal LM with KV-cache compression hooks.

    Workflow per call:
      1. Prefill the prompt with ``output_attentions=True`` to get attention
         weights and the full past_key_values.
      2. Compute per-token importance scores.
      3. Run a compression policy to map each token to FP16/INT8/DROP.
      4. Apply the actions to the cache and remember the original positions
         that survived (for ``position_ids``).
      5. Either sample/teacher-force the continuation, or just return the
         compressed state for downstream measurement.
    """

    def __init__(self, model, tokenizer, config: CompressionConfig | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or CompressionConfig()
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor):
        """Run the prompt through the model with attention output enabled."""
        out = self.model(
            input_ids=input_ids.to(self.device),
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )
        return out

    def compute_scores(
        self,
        attentions: Sequence[torch.Tensor],
        current_t: int,
    ) -> torch.Tensor:
        """[batch, seq_len] importance scores for each cached position."""
        return compute_importance(
            attentions, current_t=current_t, **self.config.importance_kwargs()
        )

    @torch.no_grad()
    def compress(
        self,
        prompt_ids: torch.Tensor,
        policy_fn: PolicyFn,
        budget: int,
    ) -> tuple:
        """Prefill + compress.

        We split the prompt into ``prompt[:-1]`` (which goes through compression)
        and ``prompt[-1]`` (the bridge token re-fed through the compressed cache
        to produce a *fair* first-step prediction). Without this split every
        policy would borrow the full-cache prefill logits for the first
        generated token.

        Returns ``(bridge_logits, bridge_past, kept_indices, stats)`` where
        ``bridge_logits`` is the next-token distribution computed AFTER the
        bridge token has been fed through the compressed cache (so it reflects
        the compressed state) and ``bridge_past`` is the corresponding cache
        ready for further decoding.
        """
        prompt_ids = prompt_ids.to(self.device)
        prefill_out = self.prefill(prompt_ids)
        return self._compress_from_prefill(
            prefill_out, prompt_ids, policy_fn, budget
        )

    @torch.no_grad()
    def _compress_from_prefill(
        self,
        prefill_out,
        prompt_ids: torch.Tensor,
        policy_fn: PolicyFn,
        budget: int,
    ) -> tuple:
        """Variant that reuses an existing prefill output (computed with
        ``output_attentions=True`` over the full prompt)."""
        prompt_ids = prompt_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]
        past_full = _to_legacy(prefill_out.past_key_values)
        attentions = prefill_out.attentions

        # Score every position over the full prompt (so the last token's
        # attention contribution is taken into account when scoring earlier
        # positions). We only compress positions ``[0, prompt_len-1)`` because
        # the last token will be re-fed.
        scores = self.compute_scores(attentions, current_t=prompt_len)
        head_scores = scores[0, :-1]
        actions = policy_fn(head_scores, budget)

        # Slice past so it covers only the first prompt_len-1 tokens.
        past_head = tuple(
            (k[:, :, :-1, :], v[:, :, :-1, :]) for k, v in past_full
        )
        compressed_head, kept = apply_compression(past_head, actions)

        # Bridge step: feed the last prompt token through the compressed cache.
        bridge_token = prompt_ids[:, -1:]
        pos_ids = torch.tensor([[prompt_len - 1]], device=self.device)
        bridge_out = self.model(
            input_ids=bridge_token,
            past_key_values=_to_cache(compressed_head),
            position_ids=pos_ids,
            use_cache=True,
            return_dict=True,
        )
        bridge_logits = bridge_out.logits  # [1, 1, vocab]
        bridge_past = bridge_out.past_key_values

        n_fp16 = sum(1 for a in actions if a == "FP16")
        n_int8 = sum(1 for a in actions if a == "INT8")
        n_drop = sum(1 for a in actions if a == "DROP")
        # The bridge token (last prompt token) is always kept at full precision
        # in the cache so we count it as one extra FP16 slot for accounting.
        stats = CompressionStats(
            seq_len=prompt_len,
            n_fp16=n_fp16 + 1,
            n_int8=n_int8,
            n_drop=n_drop,
            cost=count_memory_cost(actions) + 2,
            memory_ratio=(count_memory_cost(actions) + 2) / (2.0 * prompt_len),
            actions=actions + ["FP16"],
        )
        return bridge_logits, bridge_past, kept, stats

    @torch.no_grad()
    def teacher_force_logprob_with_prefill(
        self,
        prefill_out,
        prompt_ids: torch.Tensor,
        target_ids: torch.Tensor,
        policy_fn: PolicyFn,
        budget: int,
    ) -> tuple[torch.Tensor, "CompressionStats"]:
        """Variant of ``teacher_force_logprob`` that reuses an existing
        prefill output. Lets the caller share the prefill across all policies
        for the same prompt."""
        prompt_ids = prompt_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]
        target_len = target_ids.shape[1]
        if target_len < 1:
            raise ValueError("target_ids must be non-empty")

        bridge_logits, bridge_past, _, stats = self._compress_from_prefill(
            prefill_out, prompt_ids, policy_fn, budget
        )
        first_logits = bridge_logits[0, -1, :]
        first_nll = F.cross_entropy(
            first_logits.unsqueeze(0), target_ids[:, 0], reduction="none"
        )
        if target_len == 1:
            return first_nll, stats

        feed_ids = target_ids[:, :-1]
        feed_len = feed_ids.shape[1]
        positions = torch.arange(
            prompt_len, prompt_len + feed_len, device=self.device
        ).unsqueeze(0)
        out = self.model(
            input_ids=feed_ids,
            past_key_values=bridge_past,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        )
        rest_logits = out.logits[0]
        rest_nll = F.cross_entropy(
            rest_logits, target_ids[0, 1:], reduction="none"
        )
        nll = torch.cat([first_nll, rest_nll], dim=0)
        return nll, stats

    @torch.no_grad()
    def teacher_force_logprob(
        self,
        prompt_ids: torch.Tensor,
        target_ids: torch.Tensor,
        policy_fn: PolicyFn,
        budget: int,
    ) -> tuple[torch.Tensor, "CompressionStats"]:
        """Teacher-forced NLL of ``target_ids`` given the compressed prompt
        cache. We feed the bridge token first (re-feeding the last prompt
        token through the compressed cache), so the per-target predictions
        are produced from the COMPRESSED state. The first target token is
        scored from the bridge logits; the remaining targets are scored from
        a teacher-forcing pass over ``target_ids[:, :-1]``."""
        prompt_ids = prompt_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]
        target_len = target_ids.shape[1]
        if target_len < 1:
            raise ValueError("target_ids must be non-empty")

        bridge_logits, bridge_past, _, stats = self.compress(
            prompt_ids, policy_fn, budget
        )
        # First target prediction comes from bridge_logits (last position).
        first_logits = bridge_logits[0, -1, :]
        first_nll = F.cross_entropy(
            first_logits.unsqueeze(0), target_ids[:, 0], reduction="none"
        )

        if target_len == 1:
            return first_nll, stats

        # Remaining target predictions: feed target[:, :-1] through bridge_past.
        feed_ids = target_ids[:, :-1]
        feed_len = feed_ids.shape[1]
        positions = torch.arange(
            prompt_len, prompt_len + feed_len, device=self.device
        ).unsqueeze(0)
        out = self.model(
            input_ids=feed_ids,
            past_key_values=bridge_past,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        )
        rest_logits = out.logits[0]  # [feed_len, vocab]
        rest_nll = F.cross_entropy(
            rest_logits, target_ids[0, 1:], reduction="none"
        )
        nll = torch.cat([first_nll, rest_nll], dim=0)
        return nll, stats

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        policy_fn: PolicyFn,
        budget: int,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, "CompressionStats"]:
        """Greedy (or temperature-sampled) decoding with the compressed cache.

        The first generated token is produced from the *bridge* logits (i.e.
        the last prompt token re-fed through the compressed cache), so all
        policies face the same compression-induced bias from token 1.
        """
        prompt_ids = prompt_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]

        bridge_logits, bridge_past, _, stats = self.compress(
            prompt_ids, policy_fn, budget
        )
        next_logits = bridge_logits[:, -1, :]
        if do_sample:
            probs = torch.softmax(next_logits / max(temperature, 1e-6), dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        else:
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
        generated = [next_tok]

        cur_past = bridge_past
        for step in range(1, max_new_tokens):
            position = prompt_len + step - 1
            position_ids = torch.tensor([[position]], device=self.device)
            out = self.model(
                input_ids=next_tok,
                past_key_values=cur_past,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            cur_past = out.past_key_values
            next_logits = out.logits[:, -1, :]
            if do_sample:
                probs = torch.softmax(next_logits / max(temperature, 1e-6), dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = next_logits.argmax(dim=-1, keepdim=True)
            generated.append(next_tok)
            if self.tokenizer.eos_token_id is not None and (
                next_tok.item() == self.tokenizer.eos_token_id
            ):
                break

        out_ids = torch.cat(generated, dim=1)
        return out_ids, stats

    @torch.no_grad()
    def generate_lazy(
        self,
        prompt_ids: torch.Tensor,
        policy_fn: PolicyFn,
        budget: int,
        max_new_tokens: int = 64,
        delta: int = 16,
        drift_threshold: float | None = 0.3,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """Greedy decoding with *lazy* compression rebalances.

        Initial state is built by the same prefill+bridge path as
        ``generate``. During decoding we:
          1. accumulate per-position attention received from new queries;
          2. track query drift via the cosine distance of the last-layer
             hidden state against the snapshot taken at the last rebalance;
          3. every ``delta`` steps (or when ``drift > drift_threshold``)
             re-score every cached slot, re-run the policy and re-apply
             compression to the live cache.

        Returns ``(generated_ids, info)``. ``info`` reports rebalance events,
        per-step ms and the initial compression stats."""
        import time
        from .compress import apply_compression
        prompt_ids = prompt_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]

        prefill_out = self.prefill(prompt_ids)
        prompt_attn_received = self.compute_attention_received_from_prefill(
            prefill_out.attentions
        )

        bridge_logits, cur_past, _, stats0 = self._compress_from_prefill(
            prefill_out, prompt_ids, policy_fn, budget,
        )
        info: dict = {
            "initial_stats": stats0,
            "rebalances": [],
            "per_step_ms": [],
        }

        # score_per_cached tracks accumulated importance for each cache slot
        # (corresponds to the *current* compressed cache, in order).
        kept_indices = [i for i, a in enumerate(stats0.actions) if a != "DROP"]
        score_per_cached: list[float] = [
            float(prompt_attn_received[i].item()) for i in kept_indices
        ]
        # Append the bridge token (always present).
        score_per_cached.append(float(prompt_attn_received[prompt_len - 1].item()))

        # Drift reference: last-layer hidden state of the bridge step.
        last_hidden_ref = self._safe_last_hidden(prefill_out)
        if last_hidden_ref is not None:
            last_hidden_ref = last_hidden_ref[0, -1].detach()

        next_logits = bridge_logits[:, -1, :]
        next_tok = (
            torch.multinomial(
                torch.softmax(next_logits / max(temperature, 1e-6), dim=-1), 1
            ) if do_sample else next_logits.argmax(dim=-1, keepdim=True)
        )
        generated = [next_tok]
        steps_since_rebalance = 0

        for step in range(1, max_new_tokens):
            t_step = time.perf_counter()
            position = prompt_len + step - 1
            pos_ids = torch.tensor([[position]], device=self.device)
            out = self.model(
                input_ids=next_tok,
                past_key_values=cur_past,
                position_ids=pos_ids,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=(drift_threshold is not None),
                return_dict=True,
            )
            cur_past = out.past_key_values

            # Accumulate per-slot attention.
            cur_cache_len = (
                cur_past.get_seq_length()
                if hasattr(cur_past, "get_seq_length")
                else cur_past[0][0].shape[2]
            )
            while len(score_per_cached) < cur_cache_len:
                score_per_cached.append(0.0)
            step_acc = torch.zeros(cur_cache_len, device=self.device)
            n_layers_used = 0
            for attn in out.attentions:
                if attn is None:
                    continue
                step_acc[: attn.shape[-1]] += attn[0].sum(dim=(0, 1))
                n_layers_used += 1
            if n_layers_used > 0:
                step_acc /= n_layers_used
            step_acc_cpu = step_acc.cpu().tolist()
            for i in range(cur_cache_len):
                score_per_cached[i] += step_acc_cpu[i]

            # Drift.
            trigger_drift = False
            cos_val = None
            if drift_threshold is not None and last_hidden_ref is not None:
                hidden = self._safe_last_hidden(out)
                if hidden is not None:
                    cur_vec = hidden[0, -1].detach().float()
                    ref = last_hidden_ref.float()
                    cos = torch.nn.functional.cosine_similarity(
                        cur_vec.unsqueeze(0), ref.unsqueeze(0), dim=-1
                    ).item()
                    cos_val = cos
                    if 1.0 - cos > drift_threshold:
                        trigger_drift = True

            steps_since_rebalance += 1
            trigger_delta = steps_since_rebalance >= delta

            do_rebalance = (
                (trigger_delta or trigger_drift)
                and step < max_new_tokens - 1
                and cur_cache_len > 0
            )
            if do_rebalance:
                seq_len_cached = len(score_per_cached)
                scores_tensor = torch.tensor(
                    score_per_cached, device=self.device, dtype=torch.float32
                )
                rng = (scores_tensor.max() - scores_tensor.min()).clamp(min=1e-8)
                attn_norm = (scores_tensor - scores_tensor.min()) / rng

                cfg = self.config
                rec_raw = torch.exp(-cfg.lambd * torch.arange(
                    seq_len_cached - 1, -1, -1, device=self.device,
                    dtype=torch.float32,
                ))
                rec_rng = (rec_raw.max() - rec_raw.min()).clamp(min=1e-8)
                rec_norm = (rec_raw - rec_raw.min()) / rec_rng

                pre = torch.zeros(seq_len_cached, device=self.device)
                pre[: min(cfg.prefix_len, seq_len_cached)] = 1.0

                score = (
                    (cfg.alpha if cfg.use_attention else 0.0) * attn_norm
                    + (cfg.beta if cfg.use_recency else 0.0) * rec_norm
                    + (cfg.gamma if cfg.use_prefix else 0.0) * pre
                )

                actions = policy_fn(score, budget)
                past_legacy = _to_legacy(cur_past)
                compressed_past, kept = apply_compression(past_legacy, actions)
                cur_past = _to_cache(compressed_past)

                old_len = seq_len_cached
                new_len = len(kept)
                score_per_cached = [score_per_cached[i] for i in kept]
                info["rebalances"].append({
                    "step": step,
                    "trigger": "drift" if trigger_drift else "delta",
                    "cos_distance": (1 - cos_val) if cos_val is not None else None,
                    "old_cache_len": old_len,
                    "new_cache_len": new_len,
                    "actions_summary": {
                        "FP16": sum(1 for a in actions if a == "FP16"),
                        "INT8": sum(1 for a in actions if a == "INT8"),
                        "DROP": sum(1 for a in actions if a == "DROP"),
                    },
                })
                steps_since_rebalance = 0
                ref_hidden = self._safe_last_hidden(out)
                if ref_hidden is not None:
                    last_hidden_ref = ref_hidden[0, -1].detach()

            next_logits = out.logits[:, -1, :]
            next_tok = (
                torch.multinomial(
                    torch.softmax(next_logits / max(temperature, 1e-6), dim=-1), 1
                ) if do_sample else next_logits.argmax(dim=-1, keepdim=True)
            )
            generated.append(next_tok)
            info["per_step_ms"].append((time.perf_counter() - t_step) * 1000)

            if self.tokenizer.eos_token_id is not None and (
                next_tok.item() == self.tokenizer.eos_token_id
            ):
                break

        out_ids = torch.cat(generated, dim=1)
        return out_ids, info

    def compute_attention_received_from_prefill(self, attentions):
        """Per-position attention received, averaged across layers and heads.
        Returns a 1-D tensor of length seq_len (for the first batch element)."""
        from .importance import compute_attention_received
        return compute_attention_received(attentions)[0]

    def _safe_last_hidden(self, model_output):
        """Fetch the last hidden state from a model output, or None if absent."""
        if hasattr(model_output, "hidden_states") and model_output.hidden_states:
            return model_output.hidden_states[-1]
        return None
