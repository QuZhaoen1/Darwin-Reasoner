"""A/B/C probe: is the 39.6% baseline accuracy real, or a harness artifact?

Same 8 problems, three generation conditions, scored two ways. The point is to
separate "Qwen3-8B cannot do these" from "the harness loses answers the model
already produced".

A  raw completion prompt, 1024 tokens   -- reproduces the current pilot exactly
B  raw completion prompt, 3072 tokens   -- isolates truncation
C  chat template + EOS, 3072 tokens     -- isolates the missing chat template
"""
import json
import re
import sys

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.insert(0, "src")
from darwin_reasoner.verifier import extract_final_answer, normalize_answer

MODEL = "Qwen/Qwen3-8B"
DATA = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8

rows = [json.loads(l) for l in open(DATA, encoding="utf-8")][:N]
print(f"probing {len(rows)} problems\n", flush=True)

INSTR = "Solve the problem carefully. End with 'FINAL: <answer>'."


def raw_prompt(p):
    return f"Problem:\n{p}\n\nCurrent reasoning:\n[empty]\n\n{INSTR}"


tok = AutoTokenizer.from_pretrained(MODEL)


def chat_prompt(p):
    return tok.apply_chat_template(
        [{"role": "user", "content": f"{p}\n\n{INSTR}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def latex_loose(s):
    """Tolerant comparison: strip LaTeX wrappers that carry no mathematical content."""
    s = str(s or "")
    s = re.sub(r"\\(?:left|right|!|,|;|:)", "", s)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:d)?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = s.replace("\\pi", "pi").replace("\u03c0", "pi")
    s = s.replace("^\\circ", "deg").replace("\\circ", "deg")
    s = re.sub(r"[\\${}\s]", "", s)
    return s.lower().rstrip(".")


llm = LLM(model=MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="bfloat16", max_model_len=8192)

CONDITIONS = [
    ("A raw/1024",  raw_prompt,  1024, None),
    ("B raw/3072",  raw_prompt,  3072, None),
    ("C chat/3072", chat_prompt, 3072, None),
]

summary = []
for name, builder, max_tok, stop in CONDITIONS:
    prompts = [builder(r["prompt"] if "prompt" in r else r["question"]) for r in rows]
    params = SamplingParams(max_tokens=max_tok, temperature=0.6, top_p=0.95, seed=42, stop=stop)
    outs = llm.generate(prompts, params)

    strict = loose = 0
    n_capped = 0
    print(f"\n########## {name} ##########", flush=True)
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        n_out = len(o.outputs[0].token_ids)
        capped = n_out >= max_tok - 2
        n_capped += capped
        got = extract_final_answer(text)
        gold = r.get("answer")
        s_ok = normalize_answer(got) == normalize_answer(str(gold))
        l_ok = latex_loose(got) == latex_loose(gold)
        strict += s_ok
        loose += l_ok
        flag = "CAP" if capped else "   "
        print(f"  {flag} out={n_out:5d} strict={'Y' if s_ok else 'n'} loose={'Y' if l_ok else 'n'} "
              f"got={got[:44]!r:48s} gold={str(gold)[:28]!r}", flush=True)
    print(f"  -> strict {strict}/{len(rows)}  loose {loose}/{len(rows)}  capped {n_capped}/{len(rows)}", flush=True)
    summary.append((name, strict, loose, n_capped, len(rows)))

print("\n\n================ SUMMARY ================")
print(f"{'condition':14s} {'repo verifier':>14s} {'latex-tolerant':>15s} {'truncated':>10s}")
for name, s, l, c, n in summary:
    print(f"{name:14s} {s}/{n} = {s/n:5.0%}   {l}/{n} = {l/n:5.0%}      {c}/{n}")
