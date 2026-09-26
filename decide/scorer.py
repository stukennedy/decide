"""Decision scoring on MLX: one forward pass per option order, no generated tokens.

The model sees the context, the question and lettered options, and we read its next-token
logits for the option letters. Small models prefer some letters and positions regardless
of content, so `orders` > 1 re-scores the same decision with the options permuted and
averages each option's log-probability across the permutations. The context and question
come before the options in the prompt, so every permutation shares that prefix: it's
computed once and only the options part is re-run.

The letter-readout approach follows SemIf (github.com/TheoLeeCJ/openjev, MIT).
"""
import copy
import json
import math
import time

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

PROMPTS = {
    # structured JSON payload, the layout SemIf uses. The default: best on JevBench (186/231).
    "json": {
        "system": ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
                   "Respond with only its uppercase letter, with no explanation or reasoning."),
        "user": lambda state, question, lettered: json.dumps(
            {"evidence": state, "criterion": question,
             "options": [{"letter": k, "description": d} for k, d in lettered]}, ensure_ascii=False),
    },
    # experiment: plain text. Best on the dev set (78% vs 71%) but worse on JevBench (176/231);
    # see bench/RESULTS.md
    "text": {
        "system": ("You make one decision at a time. Read the context, apply the question to it, and pick "
                   "the single best option. Use only what the context establishes. "
                   "Reply with the option's letter only."),
        "user": lambda state, question, lettered: (
            f"Context:\n{state}\n\nQuestion: {question}\n\nOptions:\n"
            + "\n".join(f"{k}. {d}" for k, d in lettered) + "\n\nAnswer with one letter."),
    },
    # experiment: question before the context
    "text_qfirst": {
        "system": ("You make one decision at a time. Read the question, then the context, and pick "
                   "the single best option. Use only what the context establishes. "
                   "Reply with the option's letter only."),
        "user": lambda state, question, lettered: (
            f"Question: {question}\n\nContext:\n{state}\n\nOptions:\n"
            + "\n".join(f"{k}. {d}" for k, d in lettered) + "\n\nAnswer with one letter."),
    },
    # experiment: no system message, instructions in the user turn
    "text_nosys": {
        "system": None,
        "user": lambda state, question, lettered: (
            "Read the context, apply the question to it, and pick the single best option. "
            "Use only what the context establishes.\n\n"
            f"Context:\n{state}\n\nQuestion: {question}\n\nOptions:\n"
            + "\n".join(f"{k}. {d}" for k, d in lettered) + "\n\nReply with the option's letter only."),
    },
}


def _state_text(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def permutations(n, orders):
    """Option orders to score: identity, then reversed, then further cyclic shifts."""
    ident = list(range(n))
    perms = [ident]
    if orders >= 2:
        perms.append(ident[::-1])
    shift = 1
    while len(perms) < min(orders, 2 * n) and shift < n:
        for base in (ident, ident[::-1]):
            p = base[shift:] + base[:shift]
            if p not in perms and len(perms) < orders:
                perms.append(p)
        shift += 1
    return perms


def log_softmax(values):
    m = max(values)
    z = m + math.log(sum(math.exp(v - m) for v in values))
    return [v - z for v in values]


class Scorer:
    def __init__(self, model, tokenizer, prompt="json", orders=1, max_tokens=4096):
        if prompt not in PROMPTS:
            raise ValueError(f"unknown prompt {prompt!r}")
        self.model, self.tokenizer = model, tokenizer
        self.prompt, self.orders, self.max_tokens = PROMPTS[prompt], orders, max_tokens
        self.letter_ids = []
        for letter in LETTERS:
            ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"letter {letter} is not a single token")
            self.letter_ids.append(ids[0])

    @classmethod
    def load(cls, model_id, revision, cache_limit_mib=256, bits=None, **kwargs):
        """bits=8 or 4 quantizes the weights in memory (about 4.5 GB or 2.5 GB instead of 9 GB)."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load

        if bits not in (None, 4, 8):
            raise ValueError("bits must be 4, 8 or None")
        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(cache_limit_mib * 1024 * 1024)  # MLX would otherwise hold on to most free RAM
        model, tokenizer = load(model_id, revision=revision, lazy=bits is not None)
        if bits:
            nn.quantize(model, group_size=64, bits=bits)  # before materialising, so the BF16 weights never all sit in memory
        mx.eval(model.parameters())
        model.eval()
        scorer = cls(model, tokenizer, **kwargs)
        scorer.bits = bits
        return scorer

    def _encode(self, row, order):
        options = [row["options"][i] for i in order]
        lettered = [(LETTERS[j], o["description"]) for j, o in enumerate(options)]
        messages = [{"role": "user", "content": self.prompt["user"](_state_text(row["state"]), row["question"], lettered)}]
        if self.prompt["system"]:
            messages.insert(0, {"role": "system", "content": self.prompt["system"]})
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.max_tokens:
            raise ValueError(f"{len(ids)} input tokens exceed the {self.max_tokens}-token limit; nothing is truncated")
        # the answer letter must be a clean next token after the prompt
        probe = self.tokenizer.encode(text + "A", add_special_tokens=False)
        if probe != ids + [self.letter_ids[0]]:
            raise ValueError("prompt boundary changes answer tokenization")
        return ids

    def score(self, row):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        n = len(row["options"])
        if not 2 <= n <= len(LETTERS):
            raise ValueError(f"need 2-{len(LETTERS)} options")
        started = time.perf_counter()
        perms = permutations(n, self.orders)
        encoded = [self._encode(row, p) for p in perms]
        slots = mx.array(self.letter_ids[:n])

        # shared prefix across all orders (context + question); prefill it once
        prefix = 0
        if len(encoded) > 1:
            first = encoded[0]
            while prefix < min(map(len, encoded)) - 1 and all(e[prefix] == first[prefix] for e in encoded):
                prefix += 1
        cache = None
        if prefix:
            cache = make_prompt_cache(self.model)
            self.model(mx.array([encoded[0][:prefix]]), cache=cache)
            mx.eval([c.state for c in cache])

        totals = [0.0] * n
        for perm, ids in zip(perms, encoded):
            if cache is not None:
                branch = copy.deepcopy(cache)
                logits = self.model(mx.array([ids[prefix:]]), cache=branch)[0, -1]
            else:
                logits = self.model(mx.array([ids]))[0, -1]
            lp = log_softmax(logits.astype(mx.float32)[slots].tolist())
            for position, option_index in enumerate(perm):
                totals[option_index] += lp[position]
        mean = [t / len(perms) for t in totals]
        probs = [math.exp(v) for v in log_softmax(mean)]
        return {"id": row.get("id", "q"), "option_ids": [o["id"] for o in row["options"]],
                "probabilities": probs, "input_tokens": len(encoded[0]), "orders": len(perms),
                "total_seconds": time.perf_counter() - started}
