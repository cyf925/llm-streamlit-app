from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


Q16_MIN = 0
Q16_MAX = 65536
EXPECTED_ITER_N = 25


class PrepareCircomInputError(ValueError):
    """Raised when dense input cannot be converted into circom input safely."""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as ex:
        raise PrepareCircomInputError(f"failed to decode JSON as utf-8: {path}") from ex
    except json.JSONDecodeError as ex:
        raise PrepareCircomInputError(f"invalid JSON: {path}: {ex}") from ex


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _require_top_level(data: Mapping[str, Any], required: Sequence[str]) -> None:
    missing = [k for k in required if k not in data]
    if missing:
        raise PrepareCircomInputError(f"missing required top-level fields: {', '.join(missing)}")


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as ex:
        raise PrepareCircomInputError(f"{field} must be int-like, got {value!r}") from ex


def _parse_q16(value: Any, field: str, min_value: int = Q16_MIN, max_value: int = Q16_MAX) -> str:
    i = _as_int(value, field)
    if not (min_value <= i <= max_value):
        raise PrepareCircomInputError(f"{field} out of Q16 range [{min_value}, {max_value}]: {i}")
    return str(i)


def _extract_dims(data: Mapping[str, Any]) -> Tuple[int, int, int, int, int, List[str]]:
    shape = data.get("shape", {}) or {}
    runtime = data.get("runtime", {}) or {}

    m = _as_int(shape.get("M"), "shape.M")
    k_max = _as_int(shape.get("K_MAX"), "shape.K_MAX")
    n_max = _as_int(shape.get("N_MAX"), "shape.N_MAX")
    iter_n = _as_int(shape.get("ITER_N"), "shape.ITER_N")
    k = _as_int(runtime.get("K"), "runtime.K")
    model_ids = [str(x) for x in (runtime.get("model_ids", []) or [])]

    if m <= 0 or k_max <= 0 or n_max <= 0 or iter_n <= 0:
        raise PrepareCircomInputError(
            f"invalid shape dimensions: M={m}, K_MAX={k_max}, N_MAX={n_max}, ITER_N={iter_n}"
        )
    if iter_n != EXPECTED_ITER_N:
        raise PrepareCircomInputError(f"shape.ITER_N must be {EXPECTED_ITER_N}, got {iter_n}")
    if k < 0 or k > k_max:
        raise PrepareCircomInputError(f"runtime.K must be in [0, K_MAX], got K={k}, K_MAX={k_max}")
    if len(model_ids) != m:
        raise PrepareCircomInputError(f"runtime.model_ids length {len(model_ids)} must equal M={m}")

    return m, k_max, n_max, iter_n, k, model_ids


def _validate_fixed_point(fixed_point: Mapping[str, Any]) -> None:
    if not isinstance(fixed_point, Mapping):
        raise PrepareCircomInputError("fixed_point must be an object")

    fmt = str(fixed_point.get("format", ""))
    scale_pow2 = _as_int(fixed_point.get("scale_pow2"), "fixed_point.scale_pow2")
    scale = _as_int(fixed_point.get("scale"), "fixed_point.scale")

    if fmt != "Q16":
        raise PrepareCircomInputError(f"fixed_point.format must be 'Q16', got {fmt!r}")
    if scale_pow2 != 16:
        raise PrepareCircomInputError(f"fixed_point.scale_pow2 must be 16, got {scale_pow2}")
    if scale != Q16_MAX:
        raise PrepareCircomInputError(f"fixed_point.scale must be {Q16_MAX}, got {scale}")


def _validate_params_q16(params_q16: Mapping[str, Any]) -> Dict[str, str]:
    required = [
        "t0",
        "beta",
        "gamma",
        "alpha_imp",
        "alpha_conflict",
        "cand_decay",
        "min_tau_scale",
    ]
    missing = [k for k in required if k not in params_q16]
    if missing:
        raise PrepareCircomInputError(f"params_q16 missing required fields: {', '.join(missing)}")

    return {k: _parse_q16(params_q16.get(k), f"params_q16.{k}") for k in required}


def _validate_objects_dense(
    objects_dense: Sequence[Mapping[str, Any]],
    m: int,
    k: int,
    k_max: int,
    n_max: int,
) -> Tuple[List[int], List[int], List[int]]:
    if not isinstance(objects_dense, list):
        raise PrepareCircomInputError("objects_dense must be a list")
    if len(objects_dense) != k_max:
        raise PrepareCircomInputError(f"objects_dense length {len(objects_dense)} must equal K_MAX={k_max}")

    fact_count_by_object: List[int] = []
    is_effective_by_object: List[int] = []
    top1_choice_flat: List[int] = []

    for o, obj in enumerate(objects_dense):
        if not isinstance(obj, Mapping):
            raise PrepareCircomInputError(f"objects_dense[{o}] must be an object")

        o_idx = _as_int(obj.get("o"), f"objects_dense[{o}].o")
        if o_idx != o:
            raise PrepareCircomInputError(f"objects_dense[{o}].o must be {o}, got {o_idx}")

        is_effective = bool(obj.get("is_effective", False))
        expected_effective = o < k
        if is_effective != expected_effective:
            raise PrepareCircomInputError(
                f"objects_dense[{o}].is_effective={is_effective} inconsistent with runtime.K={k}"
            )

        fact_count = _as_int(obj.get("fact_count"), f"objects_dense[{o}].fact_count")
        if fact_count < 0 or fact_count > n_max:
            raise PrepareCircomInputError(
                f"objects_dense[{o}].fact_count={fact_count} must be in [0, {n_max}]"
            )
        if expected_effective and fact_count < 1:
            raise PrepareCircomInputError(
                f"objects_dense[{o}] is effective but fact_count={fact_count}; expected >=1"
            )
        if (not expected_effective) and fact_count != 0:
            raise PrepareCircomInputError(
                f"objects_dense[{o}] is padding object but fact_count={fact_count}; expected 0"
            )

        facts_padded = obj.get("facts_padded")
        if not isinstance(facts_padded, list) or len(facts_padded) != n_max:
            raise PrepareCircomInputError(
                f"objects_dense[{o}].facts_padded must be list length N_MAX={n_max}"
            )

        top1_by_model = obj.get("top1_by_model")
        if not isinstance(top1_by_model, list) or len(top1_by_model) != m:
            raise PrepareCircomInputError(
                f"objects_dense[{o}].top1_by_model must be list length M={m}"
            )

        for w, idx_raw in enumerate(top1_by_model):
            idx = _as_int(idx_raw, f"objects_dense[{o}].top1_by_model[{w}]")
            if idx != -1 and not (0 <= idx < fact_count):
                raise PrepareCircomInputError(
                    f"objects_dense[{o}].top1_by_model[{w}]={idx} invalid for fact_count={fact_count}"
                )
            if not expected_effective and idx != -1:
                raise PrepareCircomInputError(
                    f"objects_dense[{o}] padding object must have top1=-1, got {idx} at model {w}"
                )
            top1_choice_flat.append(idx)

        fact_count_by_object.append(fact_count)
        is_effective_by_object.append(1 if is_effective else 0)

    return fact_count_by_object, is_effective_by_object, top1_choice_flat


def _validate_dense_q16(
    dense_q16: Mapping[str, Any],
    m: int,
    k_max: int,
    n_max: int,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    dep_avg = dense_q16.get("dep_avg")
    support = dense_q16.get("support")
    imp = dense_q16.get("imp_weight")
    conf = dense_q16.get("conf_weight")

    if not isinstance(dep_avg, list) or len(dep_avg) != m:
        raise PrepareCircomInputError(f"dense_q16.dep_avg must be list length M={m}")
    dep_avg_out = [_parse_q16(dep_avg[w], f"dense_q16.dep_avg[{w}]") for w in range(m)]

    if not isinstance(support, list) or len(support) != k_max:
        raise PrepareCircomInputError(f"dense_q16.support must be list length K_MAX={k_max}")
    support_flat: List[str] = []
    for o in range(k_max):
        row_o = support[o]
        if not isinstance(row_o, list) or len(row_o) != n_max:
            raise PrepareCircomInputError(f"dense_q16.support[{o}] must be list length N_MAX={n_max}")
        for f in range(n_max):
            row_f = row_o[f]
            if not isinstance(row_f, list) or len(row_f) != m:
                raise PrepareCircomInputError(
                    f"dense_q16.support[{o}][{f}] must be list length M={m}"
                )
            for w in range(m):
                support_flat.append(_parse_q16(row_f[w], f"dense_q16.support[{o}][{f}][{w}]"))

    if not isinstance(imp, list) or len(imp) != k_max:
        raise PrepareCircomInputError(f"dense_q16.imp_weight must be list length K_MAX={k_max}")
    imp_flat: List[str] = []
    for o in range(k_max):
        row_o = imp[o]
        if not isinstance(row_o, list) or len(row_o) != n_max:
            raise PrepareCircomInputError(f"dense_q16.imp_weight[{o}] must be list length N_MAX={n_max}")
        for g in range(n_max):
            row_g = row_o[g]
            if not isinstance(row_g, list) or len(row_g) != n_max:
                raise PrepareCircomInputError(
                    f"dense_q16.imp_weight[{o}][{g}] must be list length N_MAX={n_max}"
                )
            for f in range(n_max):
                imp_flat.append(_parse_q16(row_g[f], f"dense_q16.imp_weight[{o}][{g}][{f}]"))

    if not isinstance(conf, list) or len(conf) != k_max:
        raise PrepareCircomInputError(f"dense_q16.conf_weight must be list length K_MAX={k_max}")
    conf_flat: List[str] = []
    for o in range(k_max):
        row_o = conf[o]
        if not isinstance(row_o, list) or len(row_o) != n_max:
            raise PrepareCircomInputError(f"dense_q16.conf_weight[{o}] must be list length N_MAX={n_max}")
        for g in range(n_max):
            row_g = row_o[g]
            if not isinstance(row_g, list) or len(row_g) != n_max:
                raise PrepareCircomInputError(
                    f"dense_q16.conf_weight[{o}][{g}] must be list length N_MAX={n_max}"
                )
            for f in range(n_max):
                conf_flat.append(_parse_q16(row_g[f], f"dense_q16.conf_weight[{o}][{g}][{f}]"))

    return dep_avg_out, support_flat, imp_flat, conf_flat




def _validate_runtime_contract(runtime: Mapping[str, Any], iter_n: int, params_meta: Mapping[str, Any]) -> None:
    support_mode = runtime.get("support_mode")
    if support_mode is None:
        raise PrepareCircomInputError("runtime.support_mode is required")
    if str(support_mode) != "soft_candidates":
        raise PrepareCircomInputError(
            f"runtime.support_mode must be 'soft_candidates', got {support_mode!r}"
        )

    if "max_iter" in params_meta and params_meta.get("max_iter") is not None:
        max_iter = _as_int(params_meta.get("max_iter"), "params_meta.max_iter")
        if max_iter != iter_n:
            raise PrepareCircomInputError(
                f"params_meta.max_iter={max_iter} does not match shape.ITER_N={iter_n}"
            )

def prepare_circom_input_from_dense(dense_data: Mapping[str, Any]) -> Dict[str, Any]:
    _require_top_level(
        dense_data,
        [
            "shape",
            "fixed_point",
            "runtime",
            "objects_dense",
            "params_q16",
            "params_meta",
            "dense_q16",
            "provenance",
            "constraints",
        ],
    )

    m, k_max, n_max, iter_n, k, model_ids = _extract_dims(dense_data)
    _validate_fixed_point(dense_data.get("fixed_point", {}) or {})
    runtime = dense_data.get("runtime", {}) or {}
    params_meta = dense_data.get("params_meta", {}) or {}
    _validate_runtime_contract(runtime, iter_n=iter_n, params_meta=params_meta)

    params_q16 = _validate_params_q16(dense_data.get("params_q16", {}) or {})

    fact_count_by_object, is_effective_by_object, top1_choice_flat = _validate_objects_dense(
        dense_data.get("objects_dense", []) or [],
        m=m,
        k=k,
        k_max=k_max,
        n_max=n_max,
    )

    dep_avg, support_flat, imp_flat, conf_flat = _validate_dense_q16(
        dense_data.get("dense_q16", {}) or {},
        m=m,
        k_max=k_max,
        n_max=n_max,
    )

    circom_arrays = {
        "K": k,
        "dep_avg": dep_avg,
        "support_flat": support_flat,
        "imp_flat": imp_flat,
        "conf_flat": conf_flat,
    }

    return {
        "shape": dense_data.get("shape"),
        "fixed_point": dense_data.get("fixed_point"),
        "runtime": {
            "K": k,
            "model_ids": model_ids,
            "session_id": str(runtime.get("session_id", "")),
            "sentence_id": str(runtime.get("sentence_id", "")),
            "input_text": str(runtime.get("input_text", "")),
            "support_mode": str(runtime.get("support_mode", "")),
        },
        "params_q16": params_q16,
        "params_meta": params_meta,
        "object_meta": {
            "fact_count_by_object": fact_count_by_object,
            "is_effective_by_object": is_effective_by_object,
            "top1_choice_flat": top1_choice_flat,
        },
        "circom_arrays": circom_arrays,
        "provenance": dense_data.get("provenance"),
        "constraints": dense_data.get("constraints"),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare circom-ready input from truthfinder_dense_input.json by validating "
            "fixed dimensions and flattening dense arrays."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to truthfinder_dense_input.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write truthfinder_circom_input.json (default: same dir as input)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input
    output_path: Path = args.output or (input_path.parent / "truthfinder_circom_input.json")

    if not input_path.exists():
        raise PrepareCircomInputError(f"input file not found: {input_path}")

    dense_data = _read_json(input_path)
    circom_data = prepare_circom_input_from_dense(dense_data)
    _write_json(output_path, circom_data)

    print(f"[prepare_circom_input] wrote circom input: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrepareCircomInputError as ex:
        print(f"[prepare_circom_input] error: {ex}")
        raise SystemExit(1)