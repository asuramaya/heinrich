"""Commit anatomy — where and how a transformer commits its answer.

The homing program (paper/heinrich_method.tex) found the readout commit: the
answer token comes to out-read a 16-word background panel at ~3/4 relative
depth, across models and families. This module holds the instruments that
dissected that commit, promoted from the July-5/6 experiment scripts:

  commit_anatomy   — locate L* (self-checking lens), attribute the write to
                     attention vs MLP by direct logit attribution, measure the
                     MLP write's neuron participation ratio; optional
                     attention decomposition (licenser / recency / sink /
                     other) that retired the attention-relation hypothesis.
  commit_neurons   — is the MLP answer-write localized or distributed?
                     Per-neuron contributions at the commit layer,
                     concentration stats, cross-prompt overlap, and the
                     answer-specific core (generic direction subtracted).
                     Paraphrase mode: per-FACT vs per-SURFACE-FORM overlap.
  attention_sink   — the canonical null-park test: huge attention on
                     position 0 with a low-norm value writes nothing;
                     routing happens via the small content attention.

Standing findings these measure against: the commit is MLP-led at ~3/4
relative depth with a diffuse (hundreds-of-neurons) write; the position-0
sink is a null park; licenser attention is a recency/sink confound.

All readouts route through heinrich.profile.readout.lens_logits (the one
blessed, self-checking lens — constitution). The default prompt pairs and
background panel are the PRE-REGISTERED protocol of the homing study, not a
fallback; override them to run new protocols.

Model structure: assumes the Llama family layout (model.model.layers,
model.model.norm, model.lm_head) — SmolLM2, Llama, Qwen2.5. Anything else
raises rather than guessing.
"""
from __future__ import annotations

import numpy as np

DEFAULT_BG = [" table", " music", " river", " doctor", " happy", " window",
              " garden", " metal", " quiet", " horse", " paper", " coffee",
              " stone", " engine", " lucky", " sugar"]
DEFAULT_NULL = " bicycle"
DEFAULT_FACTS = [("France", " Paris"), ("England", " London"),
                 ("Italy", " Rome"), ("Spain", " Madrid"),
                 ("Germany", " Berlin"), ("Japan", " Tokyo"),
                 ("China", " Beijing"), ("Russia", " Moscow")]
# Template sets from the between experiment. 'adjacent' makes the licenser
# the recency token (confound); 'separated' parks it at position 0 (sink
# confound); only 'middle' can show a genuine semantic lock.
TEMPLATES = {
    "adjacent": "The capital of {c} is",
    "separated": "{c} is a country and its capital is",
    "middle": "We recently learned that {c} has a capital called",
}
DEFAULT_PHRASINGS = ["The capital of {c} is",
                     "We recently learned that {c} has a capital called",
                     "{c}'s capital city is named",
                     "The city that serves as the capital of {c} is"]


def _llama_parts(model):
    """The (layers, final-norm weight, unembedding) of a Llama-family model."""
    try:
        layers = model.model.layers
        norm_w = model.model.norm.weight.detach()
        wu = model.lm_head.weight.detach()
    except AttributeError as e:
        raise AttributeError(
            f"{type(model).__name__} does not expose model.model.layers / "
            f"model.model.norm / model.lm_head — commit instruments support "
            f"the Llama family (SmolLM2, Llama, Qwen2.5)") from e
    return layers, norm_w, wu


def _one_id(tok, s: str) -> int:
    return tok.encode(s, add_special_tokens=False)[0]


def _commit_layer(lens, ans_id: int, bg_ids: list[int], n: int) -> int | None:
    """First layer from which the answer out-reads the whole panel
    persistently through to the end; None if the final layer never commits."""
    rank = [int(sum(1 for b in bg_ids
                    if float(lens[l][b]) > float(lens[l][ans_id])))
            for l in range(n)]
    if rank[-1] != 0:
        return None
    j = n - 1
    while j - 1 >= 0 and rank[j - 1] == 0:
        j -= 1
    return j


def _participation_ratio(v: np.ndarray) -> float:
    e = v ** 2
    return float((e.sum() ** 2) / ((e ** 2).sum() + 1e-12))


def _resolve_pairs(pairs, template: str) -> list[tuple[str, str, str]]:
    """-> [(prompt, answer, licenser), ...]. `pairs` entries may be
    (licenser, answer) tuples/lists or {prompt, answer, licenser} dicts."""
    tmpl = TEMPLATES[template]
    out = []
    for p in (pairs or DEFAULT_FACTS):
        if isinstance(p, dict):
            out.append((p["prompt"], p["answer"], p.get("licenser", "")))
        else:
            c, a = p
            out.append((tmpl.format(c=c), a, c))
    return out


def commit_anatomy(models: list[str],
                   *,
                   template: str = "middle",
                   pairs: list | None = None,
                   background: list[str] | None = None,
                   null_word: str = DEFAULT_NULL,
                   attention: bool = False,
                   device: str | None = None) -> dict:
    """Locate the readout commit and attribute it, per model.

    Per committed prompt: L* from the self-checking lens; per-layer direct
    logit attribution of each sublayer's last-token output onto the answer's
    unembedding direction (folded through the final RMSNorm gain) and onto a
    null direction (control, ~0); participation ratio of the MLP write at
    L*. With ``attention=True`` (forces eager attention), also the last
    token's attention mass on the licenser / recency token / position-0 sink
    / other, per layer — the decomposition that showed the commit does NOT
    ride an attention relation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .readout import lens_logits

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bg = background or DEFAULT_BG
    triples = _resolve_pairs(pairs, template)

    results = []
    for name in models:
        tok = AutoTokenizer.from_pretrained(name)
        kwargs = {"torch_dtype": torch.float32}
        if attention:
            kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs) \
            .eval().to(dev)
        layers, norm_w, wu = _llama_parts(model)
        n = len(layers)
        cap: dict = {}

        def mk(l, kind):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                cap[(l, kind)] = o[0, -1, :].detach()
            return hook

        def mkz(l):
            def pre(mod, inp):
                cap[(l, "z")] = inp[0][0, -1, :].detach()
            return pre

        handles = []
        for l, ly in enumerate(layers):
            handles.append(ly.self_attn.register_forward_hook(mk(l, "attn")))
            handles.append(ly.mlp.register_forward_hook(mk(l, "mlp")))
            handles.append(ly.mlp.down_proj.register_forward_pre_hook(mkz(l)))

        bg_ids = [_one_id(tok, w) for w in bg]
        null_id = _one_id(tok, null_word)

        rows = []
        for prompt, answer, licenser in triples:
            cap.clear()
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True,
                            output_attentions=attention)
            lens = lens_logits(out, model)
            ans_id = _one_id(tok, answer)
            lstar = _commit_layer(lens, ans_id, bg_ids, n)
            ua = wu[ans_id] * norm_w
            un = wu[null_id] * norm_w
            row = {
                "prompt": prompt, "answer": answer, "lstar": lstar,
                "top1_final": tok.decode([int(lens[-1].argmax())]),
                "attn_ans": [float(cap[(l, "attn")] @ ua) for l in range(n)],
                "mlp_ans": [float(cap[(l, "mlp")] @ ua) for l in range(n)],
                "attn_null": [float(cap[(l, "attn")] @ un) for l in range(n)],
                "mlp_null": [float(cap[(l, "mlp")] @ un) for l in range(n)],
            }
            if lstar is not None:
                wd = layers[lstar].mlp.down_proj.weight.detach()
                z = cap[(lstar, "z")]
                row["eff_neurons"] = _participation_ratio(
                    (z * (ua @ wd)).cpu().numpy())
            if attention:
                seq = ids.shape[1]
                toks = [tok.decode([i]) for i in ids[0].tolist()]
                hit = [j for j, t in enumerate(toks)
                       if licenser and licenser.lower() in t.lower().strip()]
                lp = hit[-1] if hit else None
                row["licenser_pos"] = lp
                row["seq"] = seq
                lic, rec, snk, oth = [], [], [], []
                for l in range(n):
                    arow = out.attentions[l][0, :, -1, :].mean(0) \
                        .cpu().numpy()
                    lic.append(float(arow[lp]) if lp is not None
                               else float("nan"))
                    rec.append(float(arow[seq - 2]))
                    snk.append(float(arow[0]))
                    mask = np.ones(seq, bool)
                    for k in (lp, seq - 2, 0, seq - 1):
                        if k is not None:
                            mask[k] = False
                    oth.append(float(arow[mask].mean()) if mask.any()
                               else float("nan"))
                row.update(lic=lic, recency=rec, sink=snk, other=oth)
            rows.append(row)

        for h in handles:
            h.remove()
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

        comm = [r for r in rows if r["lstar"] is not None]
        ls = [r["lstar"] for r in comm]
        summary = {
            "model": name, "n_layers": n,
            "intermediate": int(layers[0].mlp.down_proj.weight.shape[1])
            if comm else None,
            "committed": f"{len(comm)}/{len(rows)}",
            "rel_lstar": round(float(np.mean(ls)) / n, 3) if ls else None,
        }
        if comm:
            def at_star(key):
                return round(float(np.mean(
                    [r[key][r["lstar"]] for r in comm])), 2)
            summary.update(
                attn_ans_at_lstar=at_star("attn_ans"),
                mlp_ans_at_lstar=at_star("mlp_ans"),
                attn_null_at_lstar=at_star("attn_null"),
                mlp_null_at_lstar=at_star("mlp_null"),
                eff_neurons=round(float(np.mean(
                    [r["eff_neurons"] for r in comm])), 1),
                driver="MLP" if at_star("mlp_ans") > at_star("attn_ans")
                else "attention",
            )
            if attention:
                def at_star_nan(key):
                    vals = [r[key][r["lstar"]] for r in comm]
                    return round(float(np.nanmean(vals)), 3)
                summary.update(
                    lic_at_lstar=at_star_nan("lic"),
                    recency_at_lstar=at_star_nan("recency"),
                    sink_at_lstar=at_star_nan("sink"),
                    other_at_lstar=at_star_nan("other"),
                )
        results.append({"summary": summary, "rows": rows})

    return {"template": template, "background": bg, "null": null_word,
            "attention": attention, "models": results}


def commit_neurons(model_name: str,
                   *,
                   template: str = "middle",
                   pairs: list | None = None,
                   background: list[str] | None = None,
                   null_word: str = DEFAULT_NULL,
                   top_k: int = 10,
                   device: str | None = None) -> dict:
    """Localized or distributed? Per-neuron contributions to the answer
    readout at each prompt's commit layer: z_i * (down_proj[:, i] . u_ans).

    Reports concentration (participation ratio, #-to-80%, top-10 fraction),
    cross-prompt top-k overlap (generic capital-slot vs per-answer), null
    specificity, and the answer-SPECIFIC core (mean background direction
    subtracted — the debt from the neurons experiment).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .readout import lens_logits

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bg = background or DEFAULT_BG
    triples = _resolve_pairs(pairs, template)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32).eval().to(dev)
    layers, norm_w, wu = _llama_parts(model)
    n = len(layers)
    inter = int(layers[0].mlp.down_proj.weight.shape[1])
    zcap: dict = {}
    handles = [ly.mlp.down_proj.register_forward_pre_hook(
        (lambda L: (lambda m, i: zcap.__setitem__(L, i[0][0, -1, :].detach())))(l))
        for l, ly in enumerate(layers)]

    bg_ids = [_one_id(tok, w) for w in bg]
    u_bg = (wu[torch.tensor(bg_ids, device=dev)] * norm_w).mean(0)
    null_id = _one_id(tok, null_word)

    def concentration(c: np.ndarray) -> dict:
        pos = c[c > 0]
        tot = float(pos.sum())
        srt = np.sort(pos)[::-1]
        cum = np.cumsum(srt) / (tot + 1e-9)
        return {
            "eff_neurons": round(_participation_ratio(pos), 1),
            "n_to_80pct": int(np.searchsorted(cum, 0.80) + 1),
            "top10_frac": round(float(srt[:10].sum() / (tot + 1e-9)), 3),
        }

    rows = []
    for prompt, answer, _ in triples:
        zcap.clear()
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        lens = lens_logits(out, model)
        ans_id = _one_id(tok, answer)
        lstar = _commit_layer(lens, ans_id, bg_ids, n)
        if lstar is None:
            rows.append({"prompt": prompt, "answer": answer, "lstar": None})
            continue
        wd = layers[lstar].mlp.down_proj.weight.detach()
        z = zcap[lstar]
        u_ans = wu[ans_id] * norm_w
        u_null = wu[null_id] * norm_w
        u_spec = u_ans - u_bg
        contrib = (z * (u_ans @ wd)).cpu().numpy()
        contrib_null = (z * (u_null @ wd)).cpu().numpy()
        contrib_spec = (z * (u_spec @ wd)).cpu().numpy()
        top = np.argsort(contrib)[::-1][:top_k]
        top_spec = np.argsort(contrib_spec)[::-1][:top_k]
        rows.append({
            "prompt": prompt, "answer": answer, "lstar": int(lstar),
            "raw": concentration(contrib),
            "specific": concentration(contrib_spec),
            "top_neurons": [int(x) for x in top],
            "top_specific_neurons": [int(x) for x in top_spec],
            "top_to_answer": round(float(contrib[top].sum()), 1),
            "top_to_null": round(float(contrib_null[top].sum()), 1),
            "spec_top_to_null": round(float(contrib_null[top_spec].sum()), 1),
            "spec_raw_overlap": len(set(map(int, top))
                                    & set(map(int, top_spec))),
        })

    for h in handles:
        h.remove()
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    comm = [r for r in rows if r["lstar"] is not None]
    top_sets = [set(r["top_neurons"]) for r in comm]
    summary = {"model": model_name, "intermediate": inter,
               "committed": f"{len(comm)}/{len(rows)}", "top_k": top_k}
    if comm:
        shared = set.intersection(*top_sets)
        summary.update(
            eff_neurons_raw=round(float(np.mean(
                [r["raw"]["eff_neurons"] for r in comm])), 1),
            eff_neurons_specific=round(float(np.mean(
                [r["specific"]["eff_neurons"] for r in comm])), 1),
            distinct_top_neurons=len(set().union(*top_sets)),
            shared_by_all=len(shared),
            mean_top_to_answer=round(float(np.mean(
                [r["top_to_answer"] for r in comm])), 1),
            mean_top_to_null=round(float(np.mean(
                [r["top_to_null"] for r in comm])), 1),
            verdict=("LOCALIZED" if np.mean(
                [r["raw"]["eff_neurons"] for r in comm]) < inter * 0.1
                else "DISTRIBUTED"),
        )
    return {"summary": summary, "rows": rows,
            "template": template, "null": null_word}


def commit_paraphrase(model_name: str,
                      *,
                      layer: int,
                      facts: list | None = None,
                      phrasings: list[str] | None = None,
                      top_k: int = 10,
                      device: str | None = None) -> dict:
    """Per-FACT or per-SURFACE-FORM? At a fixed commit-band layer, the
    top answer-neurons for several phrasings of the same fact: within-fact
    overlap >> between-fact overlap means a stable lookup survives
    rewording."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .readout import lens_logits

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    facts = [tuple(f) for f in (facts or DEFAULT_FACTS[:4])]
    phrasings = phrasings or DEFAULT_PHRASINGS

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32).eval().to(dev)
    layers, norm_w, wu = _llama_parts(model)
    zcap: dict = {}
    handles = [ly.mlp.down_proj.register_forward_pre_hook(
        (lambda L: (lambda m, i: zcap.__setitem__(L, i[0][0, -1, :].detach())))(l))
        for l, ly in enumerate(layers)]

    wd = layers[layer].mlp.down_proj.weight.detach()

    def top_neurons(prompt, answer):
        zcap.clear()
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        lens_logits(out, model)  # self-check the readout path
        aid = _one_id(tok, answer)
        contrib = (zcap[layer] * ((wu[aid] * norm_w) @ wd)).cpu().numpy()
        return set(int(x) for x in np.argsort(contrib)[::-1][:top_k])

    tops = {}
    for fi, (c, a) in enumerate(facts):
        for pj, tmpl in enumerate(phrasings):
            tops[(fi, pj)] = top_neurons(tmpl.format(c=c), a)

    for h in handles:
        h.remove()
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    within = [len(tops[(fi, j)] & tops[(fi, k)])
              for fi in range(len(facts))
              for j in range(len(phrasings))
              for k in range(j + 1, len(phrasings))]
    between = [len(tops[(fi, pj)] & tops[(fk, pj)])
               for pj in range(len(phrasings))
               for fi in range(len(facts))
               for fk in range(fi + 1, len(facts))]
    cores = {}
    for fi, (c, a) in enumerate(facts):
        core = set.intersection(*[tops[(fi, pj)]
                                  for pj in range(len(phrasings))])
        cores[f"{c}->{a.strip()}"] = sorted(core)
    w, b = float(np.mean(within)), float(np.mean(between))
    return {
        "model": model_name, "layer": layer, "top_k": top_k,
        "n_facts": len(facts), "n_phrasings": len(phrasings),
        "within_fact_overlap": round(w, 2),
        "between_fact_overlap": round(b, 2),
        "per_fact_cores": cores,
        "verdict": ("PER-FACT (a stable lookup survives rewording)"
                    if w > 2 * max(b, 0.5)
                    else "wording-dependent (no stable per-fact core)"),
    }


def attention_sink(model_name: str,
                   *,
                   prompts: list[str] | None = None,
                   device: str | None = None) -> dict:
    """Null park or routing locus? Per layer: the last token's attention on
    position 0, the value-norm ratio sink/content, and the write-magnitude
    ratio (attn x |value|) sink/content. Null park = huge attention, value
    ratio < 1, write ratio < 1: the model attends there to attend to
    nothing, and routing happens via the small content attention."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    prompts = prompts or [
        "We recently learned that France has a capital called",
        "We recently learned that Japan has a capital called",
        "The capital of Italy is",
        "Grass on a summer day is usually the color",
    ]

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager",
        torch_dtype=torch.float32).eval().to(dev)
    layers, _, _ = _llama_parts(model)
    n = len(layers)
    vcap: dict = {}
    handles = [ly.self_attn.v_proj.register_forward_hook(
        (lambda L: (lambda m, i, o: vcap.__setitem__(L, o[0].detach())))(l))
        for l, ly in enumerate(layers)]

    a0_all, vr_all, wr_all = [], [], []
    for p in prompts:
        vcap.clear()
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = model(ids, output_attentions=True)
        a0, vr, wr = [], [], []
        for l in range(n):
            a = out.attentions[l][0, :, -1, :].mean(0)
            vn = vcap[l].norm(dim=-1)
            sink_a = float(a[0])
            sink_v = float(vn[0])
            cont_v = float(vn[1:].mean()) if len(vn) > 1 else 1e-9
            sink_w = sink_a * sink_v
            cont_w = float((a[1:] * vn[1:]).sum()) if len(vn) > 1 else 1e-9
            a0.append(sink_a)
            vr.append(sink_v / (cont_v + 1e-9))
            wr.append(sink_w / (cont_w + 1e-9))
        a0_all.append(a0)
        vr_all.append(vr)
        wr_all.append(wr)

    for h in handles:
        h.remove()
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    a0m = np.array(a0_all).mean(0)
    vrm = np.array(vr_all).mean(0)
    wrm = np.array(wr_all).mean(0)
    null_park = bool(a0m.mean() > 0.3 and vrm.mean() < 1.0
                     and wrm.mean() < 1.0)
    return {
        "model": model_name, "n_layers": n, "n_prompts": len(prompts),
        "per_layer": [{"layer": l, "sink_attn": round(float(a0m[l]), 3),
                       "value_ratio": round(float(vrm[l]), 3),
                       "write_ratio": round(float(wrm[l]), 3)}
                      for l in range(n)],
        "mean_sink_attn": round(float(a0m.mean()), 3),
        "mean_value_ratio": round(float(vrm.mean()), 3),
        "mean_write_ratio": round(float(wrm.mean()), 3),
        "verdict": "NULL PARK (attends there to attend to nothing)"
                   if null_park else "not a clean null park",
    }
