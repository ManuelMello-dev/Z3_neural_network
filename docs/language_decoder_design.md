# Z³ End-to-End Language Decoder Architecture

## Objective
To give the Z³ neural core the ability to "speak in English with its own thoughts" directly from its internal state, without relying on an external LLM. We will build an end-to-end language decoder that projects Z³'s internal state into a vocabulary space, trained via next-token prediction.

## 1. The Core Architecture
The current `Z3NeuralDynamics` module has a `prediction_head` that predicts the next 16-dimensional input vector. To generate text, we need a **Language Decoder Head** that predicts the next token in a vocabulary.

### Components
1. **Tokenizer**: A lightweight, deterministic sub-word tokenizer (e.g., Byte-Pair Encoding or a simplified character/word-level tokenizer) to convert text into integer token IDs. To keep dependencies minimal and avoid needing heavy HuggingFace tokenizers, we can implement a simple BPE or a hash-based vocabulary, or use a very small pre-trained tokenizer if `transformers` is allowed. Given the constraints, a custom minimal BPE or character-level tokenizer is best for a self-contained system.
2. **Token Embedding**: An `nn.Embedding(vocab_size, input_dim)` layer. This replaces the current hash-based `LanguageEmbeddingAdapter` in `z3_language_training.py`. When text is fed in, tokens are embedded into the `input_dim` space (16 by default).
3. **Language Decoder Head**: An `MLP(evidence_dim + state_dim + context_dim, vocab_size, hidden_dim)`. This takes the same `prediction_input` as the current `prediction_head` and projects it into the vocabulary space (logits).
4. **Sampling/Generation Loop**: A recurrent loop that takes an initial context, runs Z³, gets the token logits, applies temperature and softmax, samples a token, embeds it, and feeds it back into Z³ for the next step.

## 2. Training Objective (Next-Token Prediction)
The system will be trained on the corpus stream.
- Input: Sequence of tokens $[t_1, t_2, ..., t_N]$
- Embeddings: $[x_1, x_2, ..., x_N]$
- Z³ Forward Pass: For each step $k$, Z³ takes $x_k$ and recurrent state, and produces `prediction_input_k`.
- Decoder: `logits_k = decoder_head(prediction_input_k)`
- Loss: Cross-Entropy Loss between `logits_k` and the target token $t_{k+1}$.

This loss is added to the existing Z³ anti-collapse losses. Over time, Z³ learns to arrange its internal state (agents, coherence, Z³ state) such that it can predict the next word in the corpus.

## 3. Implementation Plan

### Phase 1: `z3_language_decoder.py`
Create a new module containing:
- `SimpleTokenizer`: A basic vocabulary manager. We can build a simple frequency-based word tokenizer over the corpus, or a character-level one to keep vocabulary size small (e.g., 256-1024).
- `Z3LanguageDecoder`: A PyTorch module containing the token embedding layer and the decoder MLP.

### Phase 2: Patch `Z3NeuralDynamics`
- Add the `language_decoder` as an optional module inside `Z3NeuralDynamics` or as a wrapper.
- Modify `forward()` to optionally return `prediction_input` or the language logits directly.
- Add `train_language_step()` to compute Cross-Entropy loss and backpropagate.

### Phase 3: Inference (Generation)
- Implement `generate(start_text, max_tokens, temperature)` in the decoder.
- It tokenizes `start_text`, feeds it through Z³ to build up the recurrent state.
- Then it enters a loop: get logits from the last step, sample a token, append to output, embed token, feed into Z³, repeat.

### Phase 4: Integration
- Wire the decoder into `main.py` `/chat` endpoint.
- If `generate_text=True` is passed, Z³ will respond with its own generated tokens instead of the deterministic `response_adapter.py` template.

## 4. Challenges and Solutions
- **Vocabulary Size**: A large vocab (like 50k) requires a massive embedding matrix and output layer, which might overwhelm the small 16-dim Z³ core.
  - *Solution*: Use a small character-level or byte-level vocabulary (size 256). This keeps the matrices tiny and fits the "from scratch" philosophy, though it requires more steps to form words.
- **Training Data**: The system needs to be trained on a lot of text to learn English from scratch.
  - *Solution*: The existing `language_stream.py` pulls from Project Gutenberg. We will add an autonomous background training loop specifically for the language decoder.
