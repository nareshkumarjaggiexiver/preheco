"""Calibration sweep — an embedder's operating point from OUR footage.

The third harness, beside golden.py ("did anything change?") and
run.py/compare.py ("is the count right?"): this one answers "what do this
embedder's cosines MEAN at this venue" (doc 15 §4), and it exists because
the one number everything hangs on — HECO_MATCH_THRESHOLD 0.363 — is the
SFace paper's 1:1 operating point, borrowed, never calibrated to a gate.
Swapping the embedder invalidates it entirely; even keeping SFace, the
venue moves it (the measured impostor ceiling is a per-camera number).

Inputs, both already produced by shipped machinery:
  * a labels export (`heco-labels/1`, the planner's labeller) — operator
    identities on face boxes, per frame ordinal;
  * a golden capture written with HECO_GOLDEN_EMBEDDINGS=1 — per-frame kept
    faces WITH their embeddings, in the kept-order contract.

The join is the scorecard's own: seq picks the picture, IoU picks the face,
kept-order binds the embedding. Same-identity pairs across frames are the
GENUINE distribution; cross-identity pairs are the IMPOSTOR one; the sweep
walks thresholds across both and reports the whole error curve, not just a
winner — a pack is chosen by an engineer reading distributions, signed at
standardization, never auto-adopted.

    python -m eval.sweep labels.json golden.jsonl --embedder sface-2021dec \
        --out pack.json

Output: `heco-pack/1` — the calibration pack doc 15 §4 defines, with the
distributions and curve that justify it embedded. The pack REFUSES to
propose a threshold when the distributions do not separate (overlap beyond
repair is an acquisition problem, not a threshold problem — doc 06: fix
acquisition before swapping models).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: Face-match floor for joining a labelled box to a golden kept face.
JOIN_IOU = 0.5
#: The swept range: below 0.15 nothing separates, above 0.7 nothing matches.
SWEEP_LO, SWEEP_HI, SWEEP_STEP = 0.15, 0.70, 0.005
#: A proposed threshold must beat the impostor tail by at least this margin.
SEPARATION_MARGIN = 0.02
#: Observations per identity entering the pairwise pools. A staff member seen
#: 500 times would otherwise contribute 125k genuine pairs — quadratically
#: drowning every transient guest's evidence AND the runtime (review finding).
#: Evenly-strided sampling keeps the identity's whole time span represented.
MAX_OBS_PER_IDENTITY = 40


def iou(a: dict, b: dict) -> float:
    """Intersection-over-union of two {x,y,w,h} boxes."""
    x1, y1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def kept_faces_with_embeddings(golden_record: dict) -> list[tuple[dict, list[float]]]:
    """(face, embedding) pairs for one golden frame, in the kept-order contract.

    Embeddings align 1:1 with faces whose gate == 'kept', in order — the
    same binding the scorecard verified over 300 real frames. A record
    without embeddings (captured before the flag, or a frame with no kept
    faces) contributes nothing rather than guessing.
    """
    embeddings = golden_record.get("embeddings") or []
    if not embeddings:
        return []
    faces = (golden_record.get("faces") or {}).get("faces") or []
    kept = [f for f in faces if f.get("gate") == "kept"]
    if len(kept) != len(embeddings):
        # The in-order contract broke for this record (a capture from a
        # mismatched build, a truncated line). Positional pairing would bind
        # vectors to the wrong faces — contribute nothing instead.
        return []
    return [
        (face, emb)
        for face, emb in zip(kept, embeddings)
        if isinstance(face.get("box"), dict)
    ]


def collect_observations(labels: dict, golden_lines: list[dict]) -> dict[str, list[np.ndarray]]:
    """identity -> unit embeddings, joined on (seq, IoU, kept-order).

    Only labelled faces WITH an identity participate: an anonymous box can
    join neither distribution. Staff identities participate as identities —
    a staff member's genuine/impostor statistics are the same physics.
    """
    frames_by_seq = {
        f["seq"]: [b for b in f.get("boxes", []) if b.get("kind") == "face" and b.get("identity")]
        for f in labels.get("frames", [])
    }
    out: dict[str, list[np.ndarray]] = {}
    for record in golden_lines:
        labelled = frames_by_seq.get(record.get("seq"))
        if not labelled:
            continue
        pairs = kept_faces_with_embeddings(record)
        # Greedy ONE-TO-ONE at the IoU floor: best pair first, each side used
        # once. Without this, two labelled boxes straddling one detection both
        # collected ITS embedding — the same vector under two identities, a
        # fabricated impostor pair at cosine 1.0 poisoning the whole tail
        # (review finding).
        scored = []
        for li, lab in enumerate(labelled):
            for pi, (face, _emb) in enumerate(pairs):
                v = iou(lab, face["box"])
                if v >= JOIN_IOU:
                    scored.append((v, li, pi))
        scored.sort(reverse=True)
        used_l, used_p = set(), set()
        for v, li, pi in scored:
            if li in used_l or pi in used_p:
                continue
            used_l.add(li)
            used_p.add(pi)
            vec = np.asarray(pairs[pi][1], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                out.setdefault(labelled[li]["identity"], []).append(vec / norm)
    return out


def pair_cosines(observations: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """(genuine, impostor) cosine arrays from the per-identity observations.

    Genuine: every within-identity pair (each a re-sighting the matcher must
    accept). Impostor: every cross-identity pair (each a merge the matcher
    must refuse). All pairs, not samples — at labelling scale (hundreds of
    observations) the quadratic is thousands of pairs, and the tails are the
    whole point: it is precisely the worst impostor pair that sets a floor.
    """
    ids = sorted(observations)
    capped: dict[str, np.ndarray] = {}
    for name in ids:
        vs = observations[name]
        if len(vs) > MAX_OBS_PER_IDENTITY:
            idx = np.linspace(0, len(vs) - 1, MAX_OBS_PER_IDENTITY).astype(int)
            vs = [vs[i] for i in idx]
        capped[name] = np.stack(vs)
    genuine_parts, impostor_parts = [], []
    for i, a in enumerate(ids):
        va = capped[a]
        sims = va @ va.T
        iu = np.triu_indices(len(va), k=1)
        if iu[0].size:
            genuine_parts.append(sims[iu])
        for b in ids[i + 1:]:
            impostor_parts.append((va @ capped[b].T).ravel())
    genuine = np.concatenate(genuine_parts) if genuine_parts else np.empty(0)
    impostor = np.concatenate(impostor_parts) if impostor_parts else np.empty(0)
    return genuine.astype(np.float64), impostor.astype(np.float64)


def percentiles(xs: np.ndarray) -> dict:
    """The distribution summary a pack records — tails first-class."""
    if xs.size == 0:
        return {"n": 0}
    q = lambda p: round(float(np.percentile(xs, p)), 4)  # noqa: E731
    return {
        "n": int(xs.size), "min": round(float(xs.min()), 4), "max": round(float(xs.max()), 4),
        "p01": q(1), "p05": q(5), "p10": q(10), "p50": q(50),
        "p90": q(90), "p95": q(95), "p99": q(99), "p999": q(99.9),
    }


def sweep_curve(genuine: np.ndarray, impostor: np.ndarray) -> list[dict]:
    """Error rates at every candidate threshold — the whole curve, reported.

    `missRate`: genuine pairs BELOW the threshold (a re-sighting refused —
    at a gate this mints a duplicate and INFLATES the bill).
    `falseMatchRate`: impostor pairs AT/ABOVE it (two people merged —
    DEFLATES the bill). The product decision between those two costs is
    exactly what standardization signs; the sweep only prices it.
    """
    curve = []
    for t in np.arange(SWEEP_LO, SWEEP_HI + 1e-9, SWEEP_STEP):
        t = round(float(t), 3)
        miss = float((genuine < t).mean()) if genuine.size else None
        fmr = float((impostor >= t).mean()) if impostor.size else None
        curve.append({
            "threshold": t,
            "missRate": None if miss is None else round(miss, 4),
            "falseMatchRate": None if fmr is None else round(fmr, 4),
        })
    return curve


def propose_pack(
    observations: dict[str, list[np.ndarray]],
    embedder_id: str,
    *,
    label_set: str | None = None,
    source: str | None = None,
    calibrated_at: str | None = None,
) -> dict:
    """The `heco-pack/1` document: distributions, curve, and a proposal —
    or an explicit refusal when the venue does not separate."""
    genuine, impostor = pair_cosines(observations)
    gstats, istats = percentiles(genuine), percentiles(impostor)
    curve = sweep_curve(genuine, impostor)

    pack: dict = {
        "format": "heco-pack/1",
        "embedderId": embedder_id,
        "labelSetId": label_set,
        "sourceGolden": source,
        "calibratedAt": calibrated_at,
        "identities": len(observations),
        "observations": sum(len(v) for v in observations.values()),
        "genuine": gstats,
        "impostor": istats,
        "curve": curve,
    }

    if genuine.size < 20 or impostor.size < 20:
        pack["proposal"] = None
        pack["refusal"] = (
            f"too few pairs to calibrate on (genuine {genuine.size}, impostor "
            f"{impostor.size}) — label more identity-bearing frames first"
        )
        return pack

    # The floor the worst impostor tail sets, with margin; separation exists
    # when the genuine distribution still has body above that floor.
    floor = istats["p999"] + SEPARATION_MARGIN
    if gstats["p50"] <= floor:
        pack["proposal"] = None
        pack["refusal"] = (
            f"no separation at this venue: impostor p99.9 {istats['p999']} + margin "
            f"reaches {round(floor, 4)}, above the genuine MEDIAN {gstats['p50']} — "
            "this is an acquisition problem (lighting, face size, pose), not a "
            "threshold problem. Doc 06: fix acquisition before swapping models."
        )
        return pack

    # Best total-error point at or above the impostor floor. Equal weighting
    # here — the curve is in the pack so a different cost ratio is one read.
    viable = [c for c in curve if c["threshold"] >= floor]
    if not viable:
        pack["proposal"] = None
        pack["refusal"] = (
            f"the impostor tail ({istats['p999']}) sits above the swept range "
            f"({SWEEP_HI}) — cosines this high across identities mean duplicate or "
            "mislabelled identities in the label set, not a threshold"
        )
        return pack
    best = min(viable, key=lambda c: (c["missRate"] + c["falseMatchRate"], c["threshold"]))
    threshold = best["threshold"]
    pack["proposal"] = {
        "threshold": threshold,
        # The learn bar keeps its house shape: templates enrol a margin above
        # the match cut (the measured pose-sweep rationale in match config).
        "templateConfidence": 0.05,
        # Heal/lock floors are reasoned from the measured impostor ceiling —
        # exactly how 0.45 was reasoned from 0.377, now per-venue.
        # Uncapped by 0.9: the floor's one job is clearing the worst measured
        # impostor, and a cap under it would defeat the floor exactly when it
        # matters most (review finding). 0.98 is the physical ceiling —
        # consecutive same-face frames sit ~0.99.
        "healMinCosine": round(min(0.98, istats["max"] + 0.05), 3),
        "trackLockMinCosine": round(min(0.98, istats["max"] + 0.05), 3),
        "nearmissFloor": round(max(0.0, gstats["p01"] - 0.01), 3),
        "atProposal": {"missRate": best["missRate"], "falseMatchRate": best["falseMatchRate"]},
    }
    return pack


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels", type=Path, help="heco-labels/1 export from the planner")
    ap.add_argument("golden", type=Path, help="golden JSONL captured with HECO_GOLDEN_EMBEDDINGS=1")
    ap.add_argument("--embedder", required=True, help="catalog id, e.g. sface-2021dec")
    ap.add_argument("--calibrated-at", default=None, help="ISO date stamped into the pack")
    ap.add_argument("--out", type=Path, default=None, help="write the pack here (default stdout)")
    args = ap.parse_args(argv)

    labels = json.loads(args.labels.read_text())
    if labels.get("format") != "heco-labels/1":
        raise SystemExit(f"{args.labels} is not a heco-labels/1 export")
    golden = [json.loads(line) for line in args.golden.read_text().splitlines() if line.strip()]

    observations = collect_observations(labels, golden)
    pack = propose_pack(
        observations, args.embedder,
        label_set=labels.get("set", {}).get("id"),
        source=args.golden.name,
        calibrated_at=args.calibrated_at,
    )
    text = json.dumps(pack, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"pack -> {args.out}")
    else:
        print(text)
    if pack.get("refusal"):
        # stderr, so a piped `sweep ... > pack.json` stays valid JSON.
        print(f"REFUSED: {pack['refusal']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
