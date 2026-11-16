"""Explicit fixed-weight ablations of the five logical M4 LSR slots.

Skipped fits return their input flow. Geometry residuals then become zero,
and fitted-flow channels receive raw flow. This changes learned-gate input
distributions; it is not an exact redundancy deletion or a retrained model.
"""

POLICIES = {"reference5": (1, 2, 3, 4, 5),
            "final1": (5,), "last45": (4, 5)}
SLOT_NAMES = {1: "F2_for_B_gate", 2: "F3_for_B_gate", 3: "B_for_M4_gate",
              4: "C_for_M4_gate", 5: "terminal_output"}


class SelectedLSR:
    def __init__(self, variant, official_lsr):
        self.keep = POLICIES[variant]
        self.official_lsr = official_lsr
        self.logical_calls = 0
        self.executed_slots = []

    def __call__(self, flow):
        self.logical_calls += 1
        if self.logical_calls > 5:
            raise RuntimeError("Unexpected sixth logical LSR slot")
        if self.logical_calls in self.keep:
            self.executed_slots.append(self.logical_calls)
            return self.official_lsr(flow)
        return flow

    def validate(self):
        if self.logical_calls != 5 or tuple(self.executed_slots) != self.keep:
            raise RuntimeError("LSR slot selection did not match the declared policy")
