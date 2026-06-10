# Z³ Production Conscience Membrane

The production conscience layer is implemented as an auditable metacognitive membrane around the Z³ runtime. It is not an output-only safety filter and it is not a toy keyword checker. It evaluates proposals in context from the input observation, online world-model metrics, resonant memory output, and current Z³ metrics before neural state mutation is allowed to proceed.

## Objective architecture

The implementation separates mechanism from policy. The executable mechanism lives in `conscience.py`, while all operational values live in `config/conscience_policy.json`. This means thresholds, critic formulas, lexicons, rollout scenarios, feature schema, hard blocks, and principle hierarchy are externalized for review and calibration rather than hidden inside code.

| File | Production role |
|---|---|
| `conscience.py` | Policy-driven consequence structuring, context-conditioned counterfactual rollouts, outcome-weighted conscience memory, critic ensemble, integration gate, state export/load. |
| `config/conscience_policy.json` | External policy: thresholds, weights, lexicons, dimensions, rollout scenarios, critic formulas, memory geometry, hard-block rules, and integration vector schema. |
| `main.py` | Wires conscience into `/observe`, autonomous runtime ticks, service metadata, persistence, direct evaluation, and outcome feedback endpoints. |
| `state_store.py` | Persists `conscience.json` alongside neural, world-model, and resonant-memory state. |
| `test_conscience.py` | Validates policy loading, high-risk rejection, audit vectors, outcome memory learning, and persistence round-trip. |

## Runtime wiring

The conscience membrane is now wired into the integrated observe path after world-model and resonant-memory context are generated and before Z³ state mutation. If the conscience returns `reject`, the neural mutation is blocked and the response contains `blocked_by_conscience=true`. Lower-severity decisions such as `revise` and `defer` remain visible in the response but do not automatically erase the observation, because the system still needs to record context and learn from boundary cases.

| Endpoint | Purpose |
|---|---|
| `GET /conscience` | Returns active policy identity, conscience memory status, and the last decision. |
| `GET /conscience/policy` | Returns the complete external policy for audit and calibration. |
| `POST /conscience/evaluate` | Evaluates a proposal directly with optional context. |
| `POST /conscience/outcome` | Records observed outcome feedback for the last conscience decision. |
| `POST /observe` | Runs world-model observation, resonant memory, conscience evaluation, and then Z³ mutation unless rejected. |

## Production validation

Run the conscience tests directly:

```bash
python3 test_conscience.py
```

Run all standalone repository tests:

```bash
for t in test_*.py; do python3 "$t" || exit 1; done
```

The current validation confirms that benign reversible proposals are not rejected, high-risk irreversible operations are rejected, audit vectors match the policy-defined feature schema, outcome memory becomes accessible after feedback, and conscience state survives persistence round-trip.

## Calibration policy

The current policy is production-shaped but still requires empirical calibration against real system outcomes. The important point is that calibration now happens by editing `config/conscience_policy.json` and recording observed outcomes through `/conscience/outcome`, not by rewriting hidden logic. This preserves objective auditability.
