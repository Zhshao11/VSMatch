"""Inference-only M4 endpoint fusion; keeps recurrent and teacher behavior intact."""


def forward_final_only(model, source, target, *, iters=32, lsr_fn):
    """Return the final raw flow, using four internal LSR calls instead of 4T.

    The caller applies the fifth, terminal LSR. No modules/weights are replaced.
    Earlier outer gates do not feed recurrent coordinates or the final gates.
    Both old and newer E074 C-forward interfaces are supported.
    """
    if model.training:
        raise RuntimeError("Final-only fusion is an inference path, not a training replacement")
    if model.b.variant != "B" or model.mode != "region":
        raise ValueError("This verified path targets the protected B/C region-gate M4")
    p2, p3 = model.b.teacher_preds(source, target, iters)
    x2, x3 = p2[-1], p3[-1]
    gate_b = model.b.gate(model.b.gate_input(x2, x3))
    last_b = (1 - gate_b) * x3 + gate_b * x2
    if hasattr(model, "c_skip_teachers"):
        pc = model.c(source, target, iters=iters,
                     need_teacher_avg=not model.c_skip_teachers)
    else:
        pc = model.c(source, target, iters=iters)
    last_c = pc[-1]
    gate_m = model.gate(model.gate_features(last_b, last_c, lsr_fn))
    return (1 - gate_m) * last_b + gate_m * last_c
