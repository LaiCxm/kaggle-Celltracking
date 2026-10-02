# EXP021: FOCUS3D-first unified event graph (V1)

## Status

This is the first implementation of the redesigned architecture. It is a
diagnostic/candidate-layer experiment, not a replacement submission and not a
claim of improved leaderboard score. The EXP018/0.946 pipeline remains the
production fallback.

## Scope of V1

V1 implements only the first two layers of the design:

1. **统一观测层**: retain Pilkwang center observations and FOCUS3D instance
   masks, confidence summary, volume, bounding box, and boundary quality.
2. **候选事件层**: build consensus/Pilkwang-only/FOCUS-only node proposals and
   enumerate all high-recall continuation edges and explicit parent-daughter1-
   daughter2 division triples.

V1 intentionally does not yet write a tracking graph, run ILP, run motion
relink, add gaps, or emit a competition submission. The purpose is to measure
whether FOCUS changes candidate recall before any classifier or solver can hide
that failure.

## First implementation contract

```text
image -> Pilkwang centers + FOCUS instance_map/confidence_map
      -> unified node proposals
      -> continuation candidates and division triples
      -> recall/ambiguity audit
```

`FOCUS-only` proposals are provisional. They are never silently treated as
true cells and are not written to the production graph in V1.

## Planned next stages

- V1.1: cache one FOCUS pass per frame and audit nodes/edges/triples on complete
  labeled movies with the official patched scorer.
- V2: add mask-aware continuation features while keeping the Pilkwang node set
  and candidate set fixed.
- V3: add explicit division-event variables to a single global solver; ordinary
  continuation and division compete before either event is selected.
- V4: only then test activation of high-quality FOCUS-only nodes.

## Stop rules

- If the union candidate set adds no labeled endpoint or complete division triple,
  stop the FOCUS-only node branch.
- Do not tune thresholds on the previously inspected four movies.
- Any official comparison must use complete movies and the patched scorer.
- Keep a clean `FOCUS off` control and the EXP018/0.946 production fallback.

## Code

- `EXP/EXP021/ARCHITECTURE_V1.md`: frozen design decisions, scope, and stop
  rules.
- `tools/focus_first/observations.py`: instance summaries and node proposals.
- `tools/focus_first/candidates.py`: source-independent continuation and
  division candidate enumeration.
- `tools/focus_first/runtime.py`: one-pass FOCUS runtime cache retaining both
  `instance_map` and `confidence_map`.
- `tools/focus_first/audit.py`: machine-readable candidate-layer summaries.
- `tools/focus_first/pipeline.py`: multi-frame orchestration that builds the
  source-independent candidate layer.
- `EXP/EXP021/train.py`: repository CLI compatibility entry point; it records
  that V1 has no trainable network instead of fabricating CV.
- `tools/test_focus_first.py`: CPU synthetic tests for the contract.
- `tools/run_focus_first_selfcheck.py`: dependency-free test runner for the
  current environment.
- `EXP/EXP021/config/child-exp000.yaml`: frozen initial configuration.
