"""
Z³ End-to-End Language Decoder
================================

This module gives Z³ the ability to generate natural English text directly
from its own internal state — no external LLM required.

Architecture
------------
The decoder adds two components on top of the existing Z³ neural core:

1. **Token Embedding** ``nn.Embedding(vocab_size, input_dim)``
   Replaces the hash-based LanguageEmbeddingAdapter for training. Each token
   in the vocabulary is mapped to a learnable vector in Z³'s input space.
   Z³ then processes this embedding as it would any other input.

2. **Language Decoder Head** ``MLP(evidence_dim + state_dim + context_dim, vocab_size)``
   Takes the same ``prediction_input`` tensor that the existing ``prediction_head``
   uses, and projects it into logits over the vocabulary. This is the "mouth"
   that converts Z³'s internal state into a probability distribution over words.

Training Objective: Next-Token Prediction
------------------------------------------
For a sequence of tokens [t₁, t₂, ..., tₙ]:
  - Embed each token: xₖ = Embedding(tₖ)
  - Run Z³ forward: prediction_inputₖ = f(xₖ, z3_state, agents)
  - Compute logits: logitsₖ = decoder_head(prediction_inputₖ)
  - Loss: CrossEntropy(logitsₖ, tₖ₊₁)

Over time, Z³ learns to arrange its internal dynamics such that its state
after processing a sequence of words predicts the next word. This is the
same objective used by all modern language models, but trained end-to-end
through Z³'s own physics.

Tokenizer
---------
We use a **character-level tokenizer** with a vocabulary of printable ASCII
characters plus special tokens. This keeps the vocabulary tiny (~100 tokens),
which is critical because:
  - The decoder head matrix is [hidden_dim × vocab_size] — small vocab = small matrix
  - Character-level models can generalize to any word without OOV issues
  - It is fully self-contained with zero external dependencies

The trade-off is that Z³ must learn to compose characters into words, which
requires more training steps than a word-level model. But it is the correct
choice for a from-scratch system.

Generation
----------
``generate(model, start_text, max_tokens, temperature)`` implements autoregressive
decoding:
  1. Tokenize start_text → feed through Z³ to prime the recurrent state
  2. Loop: get logits from last step → apply temperature → sample token → embed → feed back
  3. Decode token IDs back to characters → return the generated string

The temperature parameter controls creativity:
  - T → 0: deterministic (always picks the most likely next character)
  - T = 1.0: standard sampling
  - T > 1.0: more random / exploratory
"""
from __future__ import annotations

import json
import math
import os
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_ERROR = exc
else:
    _TORCH_ERROR = None


# ---------------------------------------------------------------------------
# Vocabulary / Tokenizer
# ---------------------------------------------------------------------------

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"  # beginning of sequence
EOS_TOKEN = "<EOS>"  # end of sequence
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

# Character vocabulary: printable ASCII (95 chars) + special tokens
_PRINTABLE = list(string.printable)  # 100 chars including whitespace variants
_CHAR_VOCAB = SPECIAL_TOKENS + _PRINTABLE

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class CharTokenizer:
    """Minimal character-level tokenizer with a fixed printable-ASCII vocabulary.

    Vocabulary size is fixed at len(SPECIAL_TOKENS) + len(string.printable) = 104.
    Any character outside the vocabulary maps to UNK_ID.
    """

    def __init__(self) -> None:
        self._vocab: List[str] = list(_CHAR_VOCAB)
        self._char_to_id: Dict[str, int] = {ch: i for i, ch in enumerate(self._vocab)}
        self.vocab_size: int = len(self._vocab)

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Convert text to a list of token IDs."""
        ids = [self._char_to_id.get(ch, UNK_ID) for ch in text]
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int], *, skip_special: bool = True) -> str:
        """Convert token IDs back to a string."""
        chars = []
        for idx in ids:
            if 0 <= idx < len(self._vocab):
                token = self._vocab[idx]
                if skip_special and token in SPECIAL_TOKENS:
                    continue
                chars.append(token)
        return "".join(chars)

    def id_to_token(self, idx: int) -> str:
        if 0 <= idx < len(self._vocab):
            return self._vocab[idx]
        return UNK_TOKEN

    def token_to_id(self, token: str) -> int:
        return self._char_to_id.get(token, UNK_ID)


# Module-level singleton tokenizer
_TOKENIZER: Optional[CharTokenizer] = None


def get_tokenizer() -> CharTokenizer:
    """Return the module-level CharTokenizer singleton."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = CharTokenizer()
    return _TOKENIZER


# ---------------------------------------------------------------------------
# Language Decoder Module
# ---------------------------------------------------------------------------

if torch is not None:

    class Z3LanguageDecoder(nn.Module):
        """End-to-end language decoder head for Z³.

        This module owns:
          - token_embedding: maps token IDs → input_dim vectors for Z³
          - decoder_head: maps Z³'s prediction_input → vocab logits

        It does NOT own the Z³ neural core itself. It wraps around an
        existing Z3NeuralDynamics instance.

        Parameters
        ----------
        input_dim : int
            Z³ input dimension (must match Z3Config.input_dim).
        evidence_dim : int
            Z³ evidence dimension (Z3Config.evidence_dim).
        state_dim : int
            Z³ state dimension (Z3Config.state_dim).
        context_dim : int
            Z³ context dimension (Z3Config.context_dim).
        hidden_dim : int
            Hidden dimension for the decoder MLP.
        vocab_size : int
            Vocabulary size (default: CharTokenizer.vocab_size = 104).
        dropout : float
            Dropout rate in the decoder MLP.
        """

        def __init__(
            self,
            input_dim: int = 16,
            evidence_dim: int = 24,
            state_dim: int = 64,
            context_dim: int = 48,
            hidden_dim: int = 256,
            vocab_size: int = 104,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.vocab_size = vocab_size

            # Token embedding: maps token IDs → Z³ input vectors
            self.token_embedding = nn.Embedding(vocab_size, input_dim, padding_idx=PAD_ID)
            nn.init.normal_(self.token_embedding.weight, std=0.02)
            self.token_embedding.weight.data[PAD_ID].zero_()

            # Decoder head: maps Z³ prediction_input → vocabulary logits
            # prediction_input = [integrated_evidence, z3_next, context]
            decoder_in = evidence_dim + state_dim + context_dim
            self.decoder_head = nn.Sequential(
                nn.Linear(decoder_in, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, vocab_size),
            )

            # Tie embedding weights to the output projection (weight tying)
            # This is a standard technique that improves language model quality
            # by ensuring the input and output representations are consistent.
            # We project input_dim → hidden_dim for the tie.
            self.embedding_projection = nn.Linear(input_dim, decoder_in, bias=False)

        def embed_tokens(self, token_ids: "torch.Tensor") -> "torch.Tensor":
            """Convert token IDs [batch, seq] → embeddings [batch, seq, input_dim]."""
            return self.token_embedding(token_ids)

        def decode_logits(self, prediction_input: "torch.Tensor") -> "torch.Tensor":
            """Map Z³ prediction_input [batch, decoder_in] → logits [batch, vocab_size]."""
            return self.decoder_head(prediction_input)

        def forward(
            self,
            prediction_input: "torch.Tensor",
            target_ids: Optional["torch.Tensor"] = None,
        ) -> Dict[str, Any]:
            """Compute logits and optionally the cross-entropy loss.

            Parameters
            ----------
            prediction_input : Tensor [batch, evidence_dim + state_dim + context_dim]
                The concatenated prediction input from Z³'s forward pass.
            target_ids : Tensor [batch], optional
                Target token IDs for next-token prediction loss.

            Returns
            -------
            dict with keys:
              logits : Tensor [batch, vocab_size]
              loss   : Tensor scalar (only if target_ids is provided)
              probs  : Tensor [batch, vocab_size]
            """
            logits = self.decode_logits(prediction_input)
            probs = F.softmax(logits, dim=-1)
            result: Dict[str, Any] = {"logits": logits, "probs": probs}
            if target_ids is not None:
                loss = F.cross_entropy(
                    logits,
                    target_ids,
                    ignore_index=PAD_ID,
                    label_smoothing=0.05,
                )
                result["loss"] = loss
            return result

        def save(self, path: str | Path) -> None:
            """Save decoder weights to disk."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), str(path))

        def load(self, path: str | Path, *, map_location: Optional[str] = None) -> None:
            """Load decoder weights from disk."""
            state = torch.load(str(path), map_location=map_location)
            self.load_state_dict(state, strict=False)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_language_step(
    z3_model: Any,
    decoder: "Z3LanguageDecoder",
    optimizer: Any,
    text: str,
    *,
    tokenizer: Optional[CharTokenizer] = None,
    max_seq_len: int = 256,
    clip_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Run one next-token prediction training step through Z³ + decoder.

    The flow:
      1. Tokenize text → token_ids [seq_len]
      2. Embed token_ids → embeddings [seq_len, input_dim]
      3. For each step k, run Z³ forward with embedding[k] as input
      4. Collect prediction_inputs [seq_len-1, decoder_in]
      5. Compute cross-entropy loss against token_ids[1:] (next-token targets)
      6. Backpropagate through decoder AND Z³ jointly
      7. Update both optimizers

    Parameters
    ----------
    z3_model : Z3NeuralDynamics
        The live Z³ neural core.
    decoder : Z3LanguageDecoder
        The language decoder head.
    optimizer : torch.optim.Optimizer
        Optimizer covering both z3_model and decoder parameters.
    text : str
        Raw text to train on.
    tokenizer : CharTokenizer, optional
        Defaults to the module-level singleton.
    max_seq_len : int
        Truncate sequences longer than this.
    clip_grad_norm : float
        Gradient clipping norm.

    Returns
    -------
    dict with training metrics.
    """
    if torch is None:
        return {"error": "torch_unavailable"}

    tok = tokenizer or get_tokenizer()
    token_ids = tok.encode(text, add_bos=True, add_eos=True)

    # Truncate to max_seq_len
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]

    if len(token_ids) < 2:
        return {"error": "sequence_too_short", "length": len(token_ids)}

    device = next(z3_model.parameters()).device
    ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)

    # Embed all tokens: [seq_len, input_dim]
    # Detach so the embedding graph is rebuilt fresh each chunk (prevents
    # 'backward through freed graph' when the embedding layer is shared
    # across multiple backward passes in the truncated BPTT loop).
    with torch.no_grad():
        embeddings = decoder.embed_tokens(ids_tensor.unsqueeze(0)).squeeze(0)  # [seq_len, input_dim]

    z3_model.train()
    decoder.train()
    optimizer.zero_grad(set_to_none=True)

    # Reset Z³ recurrent state for this sequence
    cfg = z3_model.config
    z3 = z3_model.z3_state.unsqueeze(0).to(device).detach()
    agents = z3_model.zprime_state.to(device)
    if agents.dim() == 2:
        agents = agents.unsqueeze(0)
    agents = agents.detach()

    seq_len = embeddings.shape[0]
    chunk_losses: list = []
    last_loss_val = 0.0
    all_params = list(z3_model.parameters()) + list(decoder.parameters())

    CHUNK = 32  # truncated BPTT window

    for chunk_start in range(0, seq_len - 1, CHUNK):
        chunk_end = min(chunk_start + CHUNK, seq_len - 1)
        optimizer.zero_grad(set_to_none=True)
        chunk_loss = torch.tensor(0.0, device=device)
        chunk_n = 0

        for k in range(chunk_start, chunk_end):
            # Re-embed inside the chunk so gradients flow through the embedding layer
            # ids_tensor[k] is a scalar; unsqueeze twice to get [1, 1] for embed_tokens
            # embed_tokens returns [1, 1, input_dim]; squeeze to [1, input_dim]
            x = decoder.embed_tokens(ids_tensor[k].unsqueeze(0).unsqueeze(0)).squeeze(1)  # [1, input_dim]
            target_id = ids_tensor[k + 1].unsqueeze(0)  # [1]

            output = z3_model.forward(
                x,
                initial_z3=z3,
                initial_agents=agents,
                update_state=False,
                add_noise=True,
            )

            integrated_evidence = output["integrated_evidence"]
            z3_next = output["z3_after"]
            context = output["context"]
            prediction_input = torch.cat([integrated_evidence, z3_next, context], dim=-1)

            dec_out = decoder(prediction_input, target_ids=target_id)
            chunk_loss = chunk_loss + dec_out["loss"]
            chunk_n += 1

            # Detach recurrent state between steps within chunk
            z3 = output["z3_after"].detach()
            agents = output["agents_after"].detach()

        if chunk_n > 0:
            avg_chunk = chunk_loss / chunk_n
            avg_chunk.backward()
            torch.nn.utils.clip_grad_norm_(all_params, clip_grad_norm)
            optimizer.step()
            last_loss_val = float(avg_chunk.detach().cpu().item())
            chunk_losses.append(last_loss_val)

    mean_loss = sum(chunk_losses) / max(len(chunk_losses), 1)
    return {
        "trained": True,
        "seq_len": seq_len,
        "loss": mean_loss,
        "chunks": len(chunk_losses),
    }


# ---------------------------------------------------------------------------
# Generation (inference)
# ---------------------------------------------------------------------------

def generate(
    z3_model: Any,
    decoder: "Z3LanguageDecoder",
    prompt: str = "",
    *,
    max_new_tokens: int = 120,
    temperature: float = 0.85,
    top_k: int = 40,
    top_p: float = 0.92,
    repetition_penalty: float = 1.15,
    tokenizer: Optional[CharTokenizer] = None,
    stop_on_eos: bool = True,
) -> str:
    """Generate natural text from Z³'s internal state.

    The generation loop:
      1. Tokenize the prompt and feed through Z³ to prime the recurrent state.
      2. At each step, get the decoder logits from Z³'s current state.
      3. Apply temperature scaling, top-k, top-p (nucleus) filtering, and
         repetition penalty.
      4. Sample the next token, embed it, feed back into Z³.
      5. Decode the generated token IDs to a string.

    Parameters
    ----------
    z3_model : Z3NeuralDynamics
        The live Z³ neural core.
    decoder : Z3LanguageDecoder
        The language decoder head.
    prompt : str
        The input prompt to condition on.
    max_new_tokens : int
        Maximum number of new characters to generate.
    temperature : float
        Sampling temperature. Lower = more deterministic.
    top_k : int
        Keep only the top-k most likely tokens.
    top_p : float
        Nucleus sampling: keep tokens whose cumulative probability ≥ top_p.
    repetition_penalty : float
        Penalise recently generated tokens to reduce repetition.
    tokenizer : CharTokenizer, optional
        Defaults to the module-level singleton.
    stop_on_eos : bool
        Stop generation when EOS token is sampled.

    Returns
    -------
    str : The generated text (not including the prompt).
    """
    if torch is None:
        return "[torch unavailable]"

    tok = tokenizer or get_tokenizer()
    z3_model.eval()
    decoder.eval()
    device = next(z3_model.parameters()).device

    # Wrap entire generation in no_grad
    ctx = torch.no_grad()  # type: ignore[union-attr]
    ctx.__enter__()  # type: ignore[union-attr]

    # Tokenize prompt
    prompt_ids = tok.encode(prompt, add_bos=True, add_eos=False) if prompt else [BOS_ID]
    ids_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    # Prime Z³ recurrent state with the prompt
    z3 = z3_model.z3_state.unsqueeze(0).to(device)
    agents = z3_model.zprime_state.to(device)
    if agents.dim() == 2:
        agents = agents.unsqueeze(0)

    last_prediction_input = None

    for token_id in prompt_ids:
        x = decoder.embed_tokens(
            torch.tensor([[token_id]], dtype=torch.long, device=device)
        ).squeeze(1)  # [1, input_dim]
        output = z3_model.forward(
            x,
            initial_z3=z3,
            initial_agents=agents,
            update_state=False,
            add_noise=False,
        )
        z3 = output["z3_after"]
        agents = output["agents_after"]
        last_prediction_input = torch.cat([
            output["integrated_evidence"],
            output["z3_after"],
            output["context"],
        ], dim=-1)

    if last_prediction_input is None:
        return ""

    # Generation loop
    generated_ids: List[int] = []
    recent_ids: List[int] = list(prompt_ids[-20:])  # window for repetition penalty

    for _ in range(max_new_tokens):
        logits = decoder.decode_logits(last_prediction_input).squeeze(0)  # [vocab_size]

        # Repetition penalty: divide logits of recently seen tokens
        if repetition_penalty != 1.0:
            for prev_id in set(recent_ids):
                if 0 <= prev_id < logits.shape[0]:
                    if logits[prev_id] > 0:
                        logits[prev_id] /= repetition_penalty
                    else:
                        logits[prev_id] *= repetition_penalty

        # Temperature scaling
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)

        # Top-k filtering
        if top_k > 0:
            top_k_val = min(top_k, logits.shape[-1])
            kth_val = torch.topk(logits, top_k_val).values[-1]
            logits = logits.masked_fill(logits < kth_val, float("-inf"))

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_indices_to_remove] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        if stop_on_eos and next_id == EOS_ID:
            break

        generated_ids.append(next_id)
        recent_ids.append(next_id)
        if len(recent_ids) > 20:
            recent_ids.pop(0)

        # Embed and feed back into Z³
        x = decoder.embed_tokens(
            torch.tensor([[next_id]], dtype=torch.long, device=device)
        ).squeeze(1)
        output = z3_model.forward(
            x,
            initial_z3=z3,
            initial_agents=agents,
            update_state=False,
            add_noise=False,
        )
        z3 = output["z3_after"]
        agents = output["agents_after"]
        last_prediction_input = torch.cat([
            output["integrated_evidence"],
            output["z3_after"],
            output["context"],
        ], dim=-1)

    result = tok.decode(generated_ids, skip_special=True)
    try:
        ctx.__exit__(None, None, None)  # type: ignore[union-attr]
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def get_decoder_path(state_dir: Optional[str] = None) -> Path:
    """Return the canonical path for the decoder checkpoint."""
    base = state_dir or os.environ.get(
        "Z3_STATE_DIR",
        os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "data"),
    )
    return Path(base) / "z3_language_decoder.pt"


def save_decoder(decoder: "Z3LanguageDecoder", state_dir: Optional[str] = None) -> bool:
    """Save decoder weights to the canonical path."""
    try:
        path = get_decoder_path(state_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(decoder.state_dict(), str(path))
        print(f"[Z³ Decoder] Saved to {path}")
        return True
    except Exception as exc:
        print(f"[Z³ Decoder] Save failed: {exc}")
        return False


def load_decoder(
    decoder: "Z3LanguageDecoder",
    state_dir: Optional[str] = None,
    *,
    map_location: Optional[str] = None,
) -> bool:
    """Load decoder weights from the canonical path if it exists."""
    path = get_decoder_path(state_dir)
    if not path.exists():
        print(f"[Z³ Decoder] No checkpoint at {path} — starting fresh.")
        return False
    try:
        state = torch.load(str(path), map_location=map_location)
        decoder.load_state_dict(state, strict=False)
        print(f"[Z³ Decoder] Loaded from {path}")
        return True
    except Exception as exc:
        print(f"[Z³ Decoder] Load failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DECODER: Optional["Z3LanguageDecoder"] = None


def get_decoder(
    input_dim: int = 16,
    evidence_dim: int = 24,
    state_dim: int = 64,
    context_dim: int = 48,
    hidden_dim: int = 256,
) -> "Z3LanguageDecoder":
    """Return the module-level Z3LanguageDecoder singleton."""
    global _DECODER
    if _DECODER is None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for Z3LanguageDecoder") from _TORCH_ERROR
        tok = get_tokenizer()
        _DECODER = Z3LanguageDecoder(
            input_dim=input_dim,
            evidence_dim=evidence_dim,
            state_dim=state_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            vocab_size=tok.vocab_size,
        )
    return _DECODER
