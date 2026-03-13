from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Circuit-semantic constants (frozen by current project circuit)
# -----------------------------------------------------------------------------
Q16 = 65536
M_FIXED = 4
K_MAX_FIXED = 15
N_MAX_FIXED = 12
ITER_N_FIXED = 25


class CircuitRefError(ValueError):
    """Raised when circom input is invalid for circuit reference execution."""


@dataclass(frozen=True)
class CircuitInput:
    shape: Mapping[str, Any]
    fixed_point: Mapping[str, Any]
    runtime: Mapping[str, Any]
    params_q16: Mapping[str, Any]
    params_meta: Mapping[str, Any]
    object_meta: Mapping[str, Any]
    circom_arrays: Mapping[str, Any]
    provenance: Mapping[str, Any]
    constraints: Mapping[str, Any]


@dataclass(frozen=True)
class ParsedArrays:
    K: int
    support: List[List[List[int]]]  # [K_MAX][N_MAX][M]
    imp: List[List[List[int]]]      # [K_MAX][N_MAX][N_MAX]  indexed [o][g][f]
    conf: List[List[List[int]]]     # [K_MAX][N_MAX][N_MAX]  indexed [o][g][f]
    dep_avg: List[int]              # [M]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as ex:
        raise CircuitRefError(f"{field} must be int-like, got {value!r}") from ex


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as ex:
        raise CircuitRefError(f"failed to decode UTF-8 JSON: {path}") from ex
    except json.JSONDecodeError as ex:
        raise CircuitRefError(f"invalid JSON: {path}: {ex}") from ex


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _require_top_level(data: Mapping[str, Any], required: Sequence[str]) -> None:
    missing = [k for k in required if k not in data]
    if missing:
        raise CircuitRefError(f"missing required top-level fields: {', '.join(missing)}")


# -----------------------------------------------------------------------------
# Q16 and circuit math helpers
# -----------------------------------------------------------------------------
def q16_mul(a: int, b: int) -> int:
    """Circuit Q16Mul semantics: floor((a*b)/65536)."""
    return (int(a) * int(b)) // Q16


def q16_clamp01(x: int) -> int:
    """Circuit-style clamp helper: clamp integer into [0, Q16]."""
    return max(0, min(int(x), Q16))


def safe_div_nonneg(num: int, den: int, fallback: int) -> int:
    """Circuit SafeDivNonNeg semantics: if den>0 floor(num/den), else fallback."""
    if den == 0:
        return int(fallback)
    return int(num) // int(den)


def argmax_with_tie_break(values: Sequence[int]) -> Tuple[int, int]:
    """
    Match circuit MaxWithTieBreak scan:
    - larger value wins
    - equal keeps previous => smaller index wins in left-to-right scan
    """
    if not values:
        raise CircuitRefError("argmax_with_tie_break requires non-empty values")

    best_idx = 0
    best_val = int(values[0])
    for i in range(1, len(values)):
        v = int(values[i])
        if best_val < v:
            best_val = v
            best_idx = i
    return best_idx, best_val


# -----------------------------------------------------------------------------
# Frozen approximation functions from truthfinder.circom
# -----------------------------------------------------------------------------
def approx_tau_q16(t_q16: int) -> int:
    """
    Frozen ApproxTauQ16 from truthfinder.circom.

    Piecewise on t in [0, 65536], then capped to 262144.
    """
    t = int(t_q16)

    if t < 16384:
        raw = q16_mul(t, 75414) + 0
    elif t < 32768:
        raw = q16_mul(t, 106290) - 7719
    elif t < 49152:
        raw = q16_mul(t, 181704) - 45426
    elif t < 57344:
        raw = q16_mul(t, 363409) - 181704
    elif t < 61440:
        raw = q16_mul(t, 726817) - 499687
    else:
        raw = q16_mul(t, 1287034) - 1024890

    return min(raw, 262144)


def approx_sigmoid_q16_signed(x_q16: int) -> int:
    """
    Frozen circuit _sigmoid_circuit semantics from truthfinder.circom.

    This implementation is STRICTLY aligned with the current final frozen Circom
    version and is the only circuit reference behavior here.

    IMPORTANT:
    - This is the circuit-semantic reference (NOT float sigmoid, NOT alternate
      signed-floor / secant interpretations).
    - Negative half-axis uses "做法 A":
        y = d - floor((c * |x|) / 65536)
    - Positive half-axis uses:
        y = d + floor((c * x) / 65536)
    - x == 0 is handled first and returns exact midpoint 32768.

    Boundary ownership (must match current Circom):
    - x=-6 -> left saturation
    - x=-4 -> (-6,-4]
    - x=-2 -> (-4,-2]
    - x=-1 -> (-2,-1]
    - x=0  -> midpoint
    - x=1  -> (0,1]
    - x=2  -> (1,2]
    - x=4  -> (2,4]
    - x=6  -> (4,6]
    """
    x = int(x_q16)

    if x == 0:
        return 32768

    # Saturation
    if x <= -393216:
        return 162
    if x >= 393216:
        return 65374

    if x < 0:
        xa = -x
        # x in (-393216, -262144] => (-6,-4]
        if xa >= 262144:
            y = 3212 - q16_mul(508, xa)
        # x in (-262144, -131072] => (-4,-2]
        elif xa >= 131072:
            y = 14445 - q16_mul(3317, xa)
        # x in (-131072, -65536] => (-2,-1]
        elif xa >= 65536:
            y = 27439 - q16_mul(9813, xa)
        # x in (-65536, 0) => (-1,0)
        else:
            y = 32768 - q16_mul(15143, xa)
    else:
        # x in (0,65536] => (0,1]
        if x <= 65536:
            y = 32768 + q16_mul(15143, x)
        # x in (65536,131072] => (1,2]
        elif x <= 131072:
            y = 38097 + q16_mul(9813, x)
        # x in (131072,262144] => (2,4]
        elif x <= 262144:
            y = 51091 + q16_mul(3317, x)
        # x in (262144,393216] => (4,6]
        else:
            y = 62324 + q16_mul(508, x)

    # Circuit clamps only upper bound to 65536; piecewise is non-negative by design.
    return min(y, Q16)


# -----------------------------------------------------------------------------
# Parsing / validation against prepare_circom_input.py output contract
# -----------------------------------------------------------------------------
def load_circuit_input(path: Path) -> CircuitInput:
    data = _read_json(path)
    _require_top_level(
        data,
        [
            "shape",
            "fixed_point",
            "runtime",
            "params_q16",
            "params_meta",
            "object_meta",
            "circom_arrays",
            "provenance",
            "constraints",
        ],
    )
    return CircuitInput(
        shape=data.get("shape", {}) or {},
        fixed_point=data.get("fixed_point", {}) or {},
        runtime=data.get("runtime", {}) or {},
        params_q16=data.get("params_q16", {}) or {},
        params_meta=data.get("params_meta", {}) or {},
        object_meta=data.get("object_meta", {}) or {},
        circom_arrays=data.get("circom_arrays", {}) or {},
        provenance=data.get("provenance", {}) or {},
        constraints=data.get("constraints", {}) or {},
    )


def _validate_shape_fixed(ci: CircuitInput) -> None:
    m = _as_int(ci.shape.get("M"), "shape.M")
    k_max = _as_int(ci.shape.get("K_MAX"), "shape.K_MAX")
    n_max = _as_int(ci.shape.get("N_MAX"), "shape.N_MAX")
    iter_n = _as_int(ci.shape.get("ITER_N"), "shape.ITER_N")

    if m != M_FIXED or k_max != K_MAX_FIXED or n_max != N_MAX_FIXED or iter_n != ITER_N_FIXED:
        raise CircuitRefError(
            "shape mismatch with frozen circuit constants: "
            f"got M={m}, K_MAX={k_max}, N_MAX={n_max}, ITER_N={iter_n}; "
            f"expected {M_FIXED}, {K_MAX_FIXED}, {N_MAX_FIXED}, {ITER_N_FIXED}"
        )

    fmt = str(ci.fixed_point.get("format", ""))
    scale_pow2 = _as_int(ci.fixed_point.get("scale_pow2"), "fixed_point.scale_pow2")
    scale = _as_int(ci.fixed_point.get("scale"), "fixed_point.scale")
    if fmt != "Q16" or scale_pow2 != 16 or scale != Q16:
        raise CircuitRefError(
            "fixed_point must be Q16/2^16/65536, got "
            f"format={fmt!r}, scale_pow2={scale_pow2}, scale={scale}"
        )


def _validate_runtime_contract(ci: CircuitInput) -> int:
    K = _as_int(ci.runtime.get("K"), "runtime.K")
    support_mode = str(ci.runtime.get("support_mode", ""))

    if K < 0 or K > K_MAX_FIXED:
        raise CircuitRefError(f"runtime.K out of range [0,{K_MAX_FIXED}]: {K}")
    if support_mode != "soft_candidates":
        raise CircuitRefError(f"runtime.support_mode must be 'soft_candidates', got {support_mode!r}")

    K_arr = _as_int(ci.circom_arrays.get("K"), "circom_arrays.K")
    if K_arr != K:
        raise CircuitRefError(f"runtime.K={K} must equal circom_arrays.K={K_arr}")

    return K


def _parse_params_q16(ci: CircuitInput) -> Dict[str, int]:
    keys = [
        "t0",
        "beta",
        "gamma",
        "alpha_imp",
        "alpha_conflict",
        "cand_decay",
        "min_tau_scale",
    ]
    out: Dict[str, int] = {}
    for k in keys:
        if k not in ci.params_q16:
            raise CircuitRefError(f"params_q16 missing field: {k}")
        out[k] = _as_int(ci.params_q16.get(k), f"params_q16.{k}")
    return out


def _parse_object_meta(ci: CircuitInput, K: int) -> Dict[str, List[int]]:
    fact_counts_raw = ci.object_meta.get("fact_count_by_object")
    is_eff_raw = ci.object_meta.get("is_effective_by_object")
    top1_raw = ci.object_meta.get("top1_choice_flat")

    if not isinstance(fact_counts_raw, list) or len(fact_counts_raw) != K_MAX_FIXED:
        raise CircuitRefError(f"object_meta.fact_count_by_object must be len={K_MAX_FIXED}")
    if not isinstance(is_eff_raw, list) or len(is_eff_raw) != K_MAX_FIXED:
        raise CircuitRefError(f"object_meta.is_effective_by_object must be len={K_MAX_FIXED}")
    if not isinstance(top1_raw, list) or len(top1_raw) != K_MAX_FIXED * M_FIXED:
        raise CircuitRefError(f"object_meta.top1_choice_flat must be len={K_MAX_FIXED * M_FIXED}")

    fact_count = [_as_int(v, f"object_meta.fact_count_by_object[{i}]") for i, v in enumerate(fact_counts_raw)]
    is_effective = [_as_int(v, f"object_meta.is_effective_by_object[{i}]") for i, v in enumerate(is_eff_raw)]
    top1_flat = [_as_int(v, f"object_meta.top1_choice_flat[{i}]") for i, v in enumerate(top1_raw)]

    # Mirror circuit metadata constraints in TruthFinderMain.
    for o in range(K_MAX_FIXED):
        expected_eff = 1 if o < K else 0
        if is_effective[o] != expected_eff:
            raise CircuitRefError(
                f"object_meta.is_effective_by_object[{o}]={is_effective[o]} inconsistent with K={K}"
            )

        fc = fact_count[o]
        if fc < 0 or fc > N_MAX_FIXED:
            raise CircuitRefError(f"fact_count_by_object[{o}] out of [0,{N_MAX_FIXED}]: {fc}")

        if expected_eff == 1 and fc == 0:
            raise CircuitRefError(f"effective object o={o} must have fact_count>0")
        if expected_eff == 0 and fc != 0:
            raise CircuitRefError(f"padding object o={o} must have fact_count=0")

    return {
        "fact_count_by_object": fact_count,
        "is_effective_by_object": is_effective,
        "top1_choice_flat": top1_flat,
    }


def _unflatten_support(flat: Sequence[Any]) -> List[List[List[int]]]:
    need = K_MAX_FIXED * N_MAX_FIXED * M_FIXED
    if len(flat) != need:
        raise CircuitRefError(f"circom_arrays.support_flat length must be {need}, got {len(flat)}")

    out = [[[0 for _ in range(M_FIXED)] for _ in range(N_MAX_FIXED)] for _ in range(K_MAX_FIXED)]
    idx = 0
    for o in range(K_MAX_FIXED):
        for f in range(N_MAX_FIXED):
            for w in range(M_FIXED):
                out[o][f][w] = _as_int(flat[idx], f"circom_arrays.support_flat[{idx}]")
                idx += 1
    return out


def _unflatten_matrix(flat: Sequence[Any], name: str) -> List[List[List[int]]]:
    need = K_MAX_FIXED * N_MAX_FIXED * N_MAX_FIXED
    if len(flat) != need:
        raise CircuitRefError(f"circom_arrays.{name} length must be {need}, got {len(flat)}")

    out = [[[0 for _ in range(N_MAX_FIXED)] for _ in range(N_MAX_FIXED)] for _ in range(K_MAX_FIXED)]
    idx = 0
    for o in range(K_MAX_FIXED):
        for g in range(N_MAX_FIXED):
            for f in range(N_MAX_FIXED):
                out[o][g][f] = _as_int(flat[idx], f"circom_arrays.{name}[{idx}]")
                idx += 1
    return out


def _parse_arrays(ci: CircuitInput, K: int) -> ParsedArrays:
    dep_raw = ci.circom_arrays.get("dep_avg")
    support_raw = ci.circom_arrays.get("support_flat")
    imp_raw = ci.circom_arrays.get("imp_flat")
    conf_raw = ci.circom_arrays.get("conf_flat")

    if not isinstance(dep_raw, list) or len(dep_raw) != M_FIXED:
        raise CircuitRefError(f"circom_arrays.dep_avg must be len={M_FIXED}")
    if not isinstance(support_raw, list):
        raise CircuitRefError("circom_arrays.support_flat must be a list")
    if not isinstance(imp_raw, list):
        raise CircuitRefError("circom_arrays.imp_flat must be a list")
    if not isinstance(conf_raw, list):
        raise CircuitRefError("circom_arrays.conf_flat must be a list")

    dep_avg = [_as_int(v, f"circom_arrays.dep_avg[{i}]") for i, v in enumerate(dep_raw)]
    support = _unflatten_support(support_raw)
    imp = _unflatten_matrix(imp_raw, "imp_flat")
    conf = _unflatten_matrix(conf_raw, "conf_flat")

    # Optional constraints compatibility checks (same spirit as prepare/expander).
    constraints = ci.constraints or {}
    if bool(constraints.get("conf_must_be_symmetric", False)):
        for o in range(K):
            for g in range(N_MAX_FIXED):
                for f in range(N_MAX_FIXED):
                    if conf[o][g][f] != conf[o][f][g]:
                        raise CircuitRefError(
                            f"conf not symmetric at object={o}, g={g}, f={f}: "
                            f"{conf[o][g][f]} != {conf[o][f][g]}"
                        )

    return ParsedArrays(K=K, support=support, imp=imp, conf=conf, dep_avg=dep_avg)


# -----------------------------------------------------------------------------
# Circuit reference execution
# -----------------------------------------------------------------------------
def run_truthfinder_circuit_ref(input_data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Execute the circuit-semantic TruthFinder reference.

    This function intentionally mirrors truthfinder.circom behavior:
    - fixed Q16 integer arithmetic
    - frozen ApproxTauQ16 / ApproxSigmoidQ16Signed
    - fixed ITER_N=25 rounds
    - support/imp/conf/dep_avg entirely from circom input
    - top1_choice_flat is compatibility metadata only (not used in core update)
    """
    ci = CircuitInput(
        shape=input_data.get("shape", {}) or {},
        fixed_point=input_data.get("fixed_point", {}) or {},
        runtime=input_data.get("runtime", {}) or {},
        params_q16=input_data.get("params_q16", {}) or {},
        params_meta=input_data.get("params_meta", {}) or {},
        object_meta=input_data.get("object_meta", {}) or {},
        circom_arrays=input_data.get("circom_arrays", {}) or {},
        provenance=input_data.get("provenance", {}) or {},
        constraints=input_data.get("constraints", {}) or {},
    )

    _validate_shape_fixed(ci)
    K = _validate_runtime_contract(ci)
    params = _parse_params_q16(ci)
    ometa = _parse_object_meta(ci, K)
    arr = _parse_arrays(ci, K)

    t0 = params["t0"]
    beta = params["beta"]
    gamma = params["gamma"]
    alpha_imp = params["alpha_imp"]
    alpha_conflict = params["alpha_conflict"]
    min_tau_scale = params["min_tau_scale"]

    fact_count_by_object = ometa["fact_count_by_object"]
    is_effective_by_object = ometa["is_effective_by_object"]

    # state initialization as in circuit main
    t_state: List[List[int]] = [[0 for _ in range(M_FIXED)] for _ in range(ITER_N_FIXED + 1)]
    s_state: List[List[List[int]]] = [
        [[0 for _ in range(N_MAX_FIXED)] for _ in range(K_MAX_FIXED)]
        for _ in range(ITER_N_FIXED + 1)
    ]

    for w in range(M_FIXED):
        t_state[0][w] = t0

    for it in range(ITER_N_FIXED):
        t_in = t_state[it]
        s_prev = s_state[it]

        # tau and dependency damping
        tau_w = [0 for _ in range(M_FIXED)]
        for w in range(M_FIXED):
            tau = approx_tau_q16(t_in[w])
            one_minus = Q16 - q16_mul(gamma, arr.dep_avg[w])
            scale_w = max(min_tau_scale, one_minus)
            tau_w[w] = q16_mul(tau, scale_w)

        # update s
        s_out = [[0 for _ in range(N_MAX_FIXED)] for _ in range(K_MAX_FIXED)]
        for o in range(K_MAX_FIXED):
            for f in range(N_MAX_FIXED):
                base_sum = 0
                for w in range(M_FIXED):
                    base_sum += q16_mul(tau_w[w], arr.support[o][f][w])

                imp_sum = 0
                conf_sum = 0
                for g in range(N_MAX_FIXED):
                    imp_sum += q16_mul(arr.imp[o][g][f], s_prev[o][g])
                    conf_sum += q16_mul(arr.conf[o][g][f], s_prev[o][g])

                imp_scaled = q16_mul(alpha_imp, imp_sum)
                conf_scaled = q16_mul(alpha_conflict, conf_sum)

                pre_score = base_sum + imp_scaled
                score = pre_score - conf_scaled

                # Circuit path: neg flag + abs + beta q16 mul + signed approx sigmoid.
                x = q16_mul(beta, abs(score))
                x_signed = -x if score < 0 else x
                sig_y = approx_sigmoid_q16_signed(x_signed)

                fact_valid = 1 if f < fact_count_by_object[o] else 0
                mask = is_effective_by_object[o] * fact_valid
                s_out[o][f] = sig_y * mask

        # update t
        t_out = [0 for _ in range(M_FIXED)]
        for w in range(M_FIXED):
            num = 0
            den = 0
            for o in range(K_MAX_FIXED):
                for f in range(N_MAX_FIXED):
                    sup = arr.support[o][f][w]
                    num += q16_mul(sup, s_out[o][f])
                    den += sup
            div = safe_div_nonneg(num, den, t_in[w])
            t_out[w] = q16_clamp01(div)

        t_state[it + 1] = t_out
        s_state[it + 1] = s_out

    t_final = t_state[ITER_N_FIXED]
    s_final = s_state[ITER_N_FIXED]

    best_model_idx, best_model_score_q16 = argmax_with_tie_break(t_final)

    winning_fact_idx_by_object: List[int] = [0 for _ in range(K_MAX_FIXED)]
    for o in range(K_MAX_FIXED):
        idx, _score = argmax_with_tie_break(s_final[o])
        eff = is_effective_by_object[o]
        win = idx * eff
        if eff == 1 and not (0 <= win < fact_count_by_object[o]):
            raise CircuitRefError(
                f"winner index out of valid range at object={o}: winner={win}, fact_count={fact_count_by_object[o]}"
            )
        if eff == 0:
            win = 0
        winning_fact_idx_by_object[o] = win

    return {
        "best_model_idx": best_model_idx,
        "best_model_score_q16": best_model_score_q16,
        "winning_fact_idx_by_object": winning_fact_idx_by_object,
        "t_final": t_final,
        "s_final": s_final,
        "meta": {
            "K": K,
            "ITER_N": ITER_N_FIXED,
            "q16": Q16,
            "support_mode": str(ci.runtime.get("support_mode", "")),
            "shape": {
                "M": M_FIXED,
                "K_MAX": K_MAX_FIXED,
                "N_MAX": N_MAX_FIXED,
                "ITER_N": ITER_N_FIXED,
            },
            "top1_choice_flat": ometa["top1_choice_flat"],
            "circuit_semantics_version": "truthfinder-circuit-spec-v1",
            "note": (
                "This output is from TruthFinder_circuit_ref.py, the circuit-semantic reference "
                "for truthfinder.circom (Q16, frozen approximations, fixed 25 rounds)."
            ),
        },
    }


def run_truthfinder_circuit_ref_from_file(input_path: str | Path) -> Dict[str, Any]:
    ci = load_circuit_input(Path(input_path))
    payload = {
        "shape": dict(ci.shape),
        "fixed_point": dict(ci.fixed_point),
        "runtime": dict(ci.runtime),
        "params_q16": dict(ci.params_q16),
        "params_meta": dict(ci.params_meta),
        "object_meta": dict(ci.object_meta),
        "circom_arrays": dict(ci.circom_arrays),
        "provenance": dict(ci.provenance),
        "constraints": dict(ci.constraints),
    }
    return run_truthfinder_circuit_ref(payload)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Circuit-semantic Python reference for truthfinder.circom. "
            "Reads truthfinder_circom_input.json, runs fixed Q16 ITER_N=25 rounds, "
            "and outputs circuit-aligned results."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to truthfinder_circom_input.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write reference output JSON",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input
    output_path: Optional[Path] = args.output

    if not input_path.exists():
        raise CircuitRefError(f"input file not found: {input_path}")

    result = run_truthfinder_circuit_ref_from_file(input_path)

    print("[TruthFinder_circuit_ref] best_model_idx:", result["best_model_idx"])
    print("[TruthFinder_circuit_ref] best_model_score_q16:", result["best_model_score_q16"])
    print("[TruthFinder_circuit_ref] winning_fact_idx_by_object:", result["winning_fact_idx_by_object"])

    if output_path is not None:
        _write_json(output_path, result)
        print(f"[TruthFinder_circuit_ref] wrote output: {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CircuitRefError as ex:
        print(f"[TruthFinder_circuit_ref] error: {ex}")
        raise SystemExit(1)
