from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


Q16_MIN = 0
Q16_MAX = 65536
TOP1_DEFAULT = -1


class ExpansionError(ValueError):
    """Raised when runtime input cannot be expanded safely."""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as ex:
        raise ExpansionError(f"failed to decode JSON as utf-8: {path}") from ex
    except json.JSONDecodeError as ex:
        raise ExpansionError(f"invalid JSON: {path}: {ex}") from ex


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _require_top_level(data: Mapping[str, Any], required: Sequence[str]) -> None:
    missing = [k for k in required if k not in data]
    if missing:
        raise ExpansionError(f"missing required top-level fields: {', '.join(missing)}")


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as ex:
        raise ExpansionError(f"{field} must be int-like, got {value!r}") from ex


def _parse_q16(value: Any, field: str, min_value: int = Q16_MIN, max_value: int = Q16_MAX) -> str:
    """
    Parse Q16 value from runtime input and normalize to string integer.
    We keep string output for stability with existing schema/builder conventions.
    """
    i = _as_int(value, field)
    if not (min_value <= i <= max_value):
        raise ExpansionError(f"{field} out of Q16 range [{min_value}, {max_value}]: {i}")
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
        raise ExpansionError(
            f"invalid shape dimensions: M={m}, K_MAX={k_max}, N_MAX={n_max}, ITER_N={iter_n}"
        )
    if k < 0 or k > k_max:
        raise ExpansionError(f"runtime.K must be in [0, K_MAX], got K={k}, K_MAX={k_max}")
    # Current project contract is fixed-model TruthFinder with M slots fully populated
    # (see app.py MODELS + schema model_ids examples). Keep this strict for consistency.
    if len(model_ids) != m:
        raise ExpansionError(f"runtime.model_ids length {len(model_ids)} must equal M={m}")

    return m, k_max, n_max, iter_n, k, model_ids


def _validate_facts(facts: Mapping[str, Any], k: int, n_max: int) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}

    for o_str, arr in facts.items():
        o = _as_int(o_str, f"facts key {o_str!r}")
        if o < 0 or o >= k:
            raise ExpansionError(f"facts contains invalid object index {o}; valid range is [0, {k-1}]")
        if not isinstance(arr, list):
            raise ExpansionError(f"facts[{o_str!r}] must be a list")
        if len(arr) > n_max:
            raise ExpansionError(f"facts[{o}] has {len(arr)} facts > N_MAX={n_max}")
        out[o] = [str(x) for x in arr]

    for o in range(k):
        if o not in out:
            raise ExpansionError(
                f"facts missing entry for effective object o={o}; required range is [0, {k-1}]"
            )

    return out


def _build_fact_counts(facts_by_o: Mapping[int, Sequence[str]], k: int) -> List[int]:
    return [len(facts_by_o.get(o, [])) for o in range(k)]


def _normalize_defaults(defaults_q16: Mapping[str, Any], m: int) -> Dict[str, Any]:
    # Keep strict contract aligned with truthfinder_runtime_input_schema.json +
    # runtime_input_builder.py output; do not guess alternate field names.
    required = ["dep_avg", "support_default", "imp_weight_default", "conf_weight_default"]
    missing = [k for k in required if k not in defaults_q16]
    if missing:
        raise ExpansionError(f"defaults_q16 missing required fields: {', '.join(missing)}")

    dep_raw = defaults_q16.get("dep_avg")
    if not isinstance(dep_raw, list):
        raise ExpansionError("defaults_q16.dep_avg must be a list")
    if len(dep_raw) != m:
        raise ExpansionError(f"defaults_q16.dep_avg length {len(dep_raw)} must equal M={m}")
    dep_avg = [_parse_q16(dep_raw[w], f"defaults_q16.dep_avg[{w}]") for w in range(m)]

    support_default = _parse_q16(defaults_q16.get("support_default"), "defaults_q16.support_default")
    imp_default = _parse_q16(defaults_q16.get("imp_weight_default"), "defaults_q16.imp_weight_default")
    conf_default = _parse_q16(defaults_q16.get("conf_weight_default"), "defaults_q16.conf_weight_default")

    return {
        "dep_avg": dep_avg,
        "support_default": support_default,
        "imp_weight_default": imp_default,
        "conf_weight_default": conf_default,
    }


def _validate_fixed_point(fixed_point: Mapping[str, Any]) -> None:
    if not isinstance(fixed_point, Mapping):
        raise ExpansionError("fixed_point must be an object")

    fmt = str(fixed_point.get("format", ""))
    scale_pow2 = _as_int(fixed_point.get("scale_pow2"), "fixed_point.scale_pow2")
    scale = _as_int(fixed_point.get("scale"), "fixed_point.scale")

    # Strong validation: downstream dense/circom pipeline is strictly Q16-based.
    if fmt != "Q16":
        raise ExpansionError(f"fixed_point.format must be 'Q16', got {fmt!r}")
    if scale_pow2 != 16:
        raise ExpansionError(f"fixed_point.scale_pow2 must be 16, got {scale_pow2}")
    if scale != Q16_MAX:
        raise ExpansionError(f"fixed_point.scale must be {Q16_MAX}, got {scale}")


def _init_dense_q16(
    m: int,
    k_max: int,
    n_max: int,
    defaults: Mapping[str, Any],
) -> Dict[str, Any]:
    support_default = str(defaults["support_default"])
    imp_default = str(defaults["imp_weight_default"])
    conf_default = str(defaults["conf_weight_default"])

    return {
        "dep_avg": list(defaults["dep_avg"]),
        "support": [
            [[support_default for _w in range(m)] for _f in range(n_max)]
            for _o in range(k_max)
        ],
        "imp_weight": [
            [[imp_default for _f in range(n_max)] for _g in range(n_max)]
            for _o in range(k_max)
        ],
        "conf_weight": [
            [[conf_default for _f in range(n_max)] for _g in range(n_max)]
            for _o in range(k_max)
        ],
    }


def _validate_top1_choice(
    top1_choice: Mapping[str, Any],
    model_ids: Sequence[str],
    k: int,
    fact_counts: Sequence[int],
) -> Dict[str, Dict[int, int]]:
    model_set = set(model_ids)
    out: Dict[str, Dict[int, int]] = {m: {} for m in model_ids}

    for model, obj_map in top1_choice.items():
        if model not in model_set:
            raise ExpansionError(f"top1_choice contains unknown model id: {model}")
        if not isinstance(obj_map, Mapping):
            raise ExpansionError(f"top1_choice[{model!r}] must be an object map")

        parsed_map: Dict[int, int] = {}
        for o_str, idx_raw in obj_map.items():
            o = _as_int(o_str, f"top1_choice[{model}] key")
            idx = _as_int(idx_raw, f"top1_choice[{model}][{o_str}]")
            if o < 0 or o >= k:
                raise ExpansionError(
                    f"top1_choice[{model}][{o}] object index out of range [0, {k-1}]"
                )
            n_o = fact_counts[o]
            if idx != -1 and not (0 <= idx < n_o):
                raise ExpansionError(
                    f"top1_choice[{model}][{o}]={idx} invalid for N_o={n_o}; expected -1 or [0, {n_o-1}]"
                )
            parsed_map[o] = idx
        out[model] = parsed_map

    return out


def _apply_patches(
    patches: Mapping[str, Any],
    dense_q16: MutableMapping[str, Any],
    fact_counts: Sequence[int],
    m: int,
    k: int,
) -> None:
    dep = patches.get("dep_avg_patch", []) or []
    support = patches.get("support_patch", []) or []
    imp = patches.get("imp_weight_patch", []) or []
    conf = patches.get("conf_weight_patch", []) or []

    if not all(isinstance(x, Mapping) for x in dep):
        raise ExpansionError("patches.dep_avg_patch must be a list of objects")
    if not all(isinstance(x, Mapping) for x in support):
        raise ExpansionError("patches.support_patch must be a list of objects")
    if not all(isinstance(x, Mapping) for x in imp):
        raise ExpansionError("patches.imp_weight_patch must be a list of objects")
    if not all(isinstance(x, Mapping) for x in conf):
        raise ExpansionError("patches.conf_weight_patch must be a list of objects")

    for i, p in enumerate(dep):
        w = _as_int(p.get("w"), f"patches.dep_avg_patch[{i}].w")
        if w < 0 or w >= m:
            raise ExpansionError(f"dep_avg_patch[{i}] has w={w} out of range [0, {m-1}]")
        dense_q16["dep_avg"][w] = _parse_q16(p.get("value"), f"patches.dep_avg_patch[{i}].value")

    for i, p in enumerate(support):
        o = _as_int(p.get("o"), f"patches.support_patch[{i}].o")
        f = _as_int(p.get("f"), f"patches.support_patch[{i}].f")
        w = _as_int(p.get("w"), f"patches.support_patch[{i}].w")
        if o < 0 or o >= k:
            raise ExpansionError(f"support_patch[{i}] o={o} out of effective object range [0, {k-1}]")
        if w < 0 or w >= m:
            raise ExpansionError(f"support_patch[{i}] w={w} out of range [0, {m-1}]")
        n_o = fact_counts[o]
        if f < 0 or f >= n_o:
            raise ExpansionError(
                f"support_patch[{i}] f={f} invalid for object o={o} with N_o={n_o}; patching padding facts is not allowed"
            )
        dense_q16["support"][o][f][w] = _parse_q16(p.get("value"), f"patches.support_patch[{i}].value")

    for key, target in (("imp_weight_patch", "imp_weight"), ("conf_weight_patch", "conf_weight")):
        arr = imp if key == "imp_weight_patch" else conf
        for i, p in enumerate(arr):
            o = _as_int(p.get("o"), f"patches.{key}[{i}].o")
            g = _as_int(p.get("g"), f"patches.{key}[{i}].g")
            f = _as_int(p.get("f"), f"patches.{key}[{i}].f")
            if o < 0 or o >= k:
                raise ExpansionError(f"{key}[{i}] o={o} out of effective object range [0, {k-1}]")
            n_o = fact_counts[o]
            if g < 0 or g >= n_o or f < 0 or f >= n_o:
                raise ExpansionError(
                    f"{key}[{i}] (g={g}, f={f}) invalid for object o={o} with N_o={n_o}; patching padding facts is not allowed"
                )
            dense_q16[target][o][g][f] = _parse_q16(p.get("value"), f"patches.{key}[{i}].value")


def _build_objects_dense(
    objects_raw: Sequence[Mapping[str, Any]],
    facts_by_o: Mapping[int, Sequence[str]],
    top1_choice: Mapping[str, Mapping[int, int]],
    model_ids: Sequence[str],
    m: int,
    k: int,
    k_max: int,
    n_max: int,
) -> List[Dict[str, Any]]:
    objects_map: Dict[int, Mapping[str, Any]] = {}
    for i, obj in enumerate(objects_raw):
        if not isinstance(obj, Mapping):
            raise ExpansionError(f"objects[{i}] must be an object")
        o = _as_int(obj.get("o"), f"objects[{i}].o")
        if o < 0 or o >= k:
            raise ExpansionError(f"objects[{i}].o={o} out of effective object range [0, {k-1}]")
        if o in objects_map:
            raise ExpansionError(f"duplicate objects entry for o={o}")
        objects_map[o] = obj

    dense: List[Dict[str, Any]] = []
    for o in range(k_max):
        if o >= k:
            dense.append(
                {
                    "o": o,
                    "keyword": "",
                    "is_effective": False,
                    "fact_count": 0,
                    "facts_padded": ["" for _ in range(n_max)],
                    "top1_by_model": [TOP1_DEFAULT for _ in range(m)],
                }
            )
            continue

        # Effective objects must be explicit in runtime_input.objects to keep
        # object metadata auditable and stable with runtime builder output.
        if o not in objects_map:
            raise ExpansionError(f"objects missing effective object entry for o={o}")
        obj_raw = objects_map[o]
        if "is_effective" in obj_raw and obj_raw.get("is_effective") is not True:
            raise ExpansionError(
                f"objects entry for o={o} has is_effective={obj_raw.get('is_effective')!r}; expected true"
            )

        keyword = str(obj_raw.get("keyword", ""))
        facts = list(facts_by_o.get(o, []))
        fact_count = len(facts)
        if fact_count > n_max:
            raise ExpansionError(f"object o={o} has fact_count={fact_count} > N_MAX={n_max}")

        facts_padded = facts + ["" for _ in range(n_max - fact_count)]

        top1 = [TOP1_DEFAULT for _ in range(m)]
        for w, model in enumerate(model_ids):
            top1[w] = int(top1_choice.get(model, {}).get(o, TOP1_DEFAULT))

        dense.append(
            {
                "o": o,
                "keyword": keyword,
                "is_effective": True,
                "fact_count": fact_count,
                "facts_padded": facts_padded,
                "top1_by_model": top1,
            }
        )

    return dense


def _validate_optional_constraints(
    dense_q16: Mapping[str, Any],
    objects_dense: Sequence[Mapping[str, Any]],
    constraints: Mapping[str, Any],
    m: int,
    k: int,
    n_max: int,
) -> None:
    support_sum_raw = constraints.get("support_row_sum_q16")
    if support_sum_raw is not None:
        expected = _as_int(support_sum_raw, "constraints.support_row_sum_q16")
        for o in range(k):
            n_o = int(objects_dense[o]["fact_count"])
            for w in range(m):
                row_sum = sum(int(dense_q16["support"][o][f][w]) for f in range(n_o))
                # Compatible with sparse input: model may have no candidates for this object -> row sum can be 0.
                if row_sum not in (0, expected):
                    raise ExpansionError(
                        f"support row sum mismatch at object={o}, model_index={w}: got {row_sum}, expected {expected} or 0"
                    )

    if bool(constraints.get("conf_must_be_symmetric", False)):
        for o in range(k):
            n_o = int(objects_dense[o]["fact_count"])
            for g in range(n_o):
                for f in range(n_o):
                    a = dense_q16["conf_weight"][o][g][f]
                    b = dense_q16["conf_weight"][o][f][g]
                    if a != b:
                        raise ExpansionError(
                            f"conf_weight is not symmetric at object={o}, g={g}, f={f}: conf[g][f]={a}, conf[f][g]={b}"
                        )


def expand_runtime_input(runtime_data: Mapping[str, Any]) -> Dict[str, Any]:
    _require_top_level(
        runtime_data,
        [
            "shape",
            "fixed_point",
            "runtime",
            "objects",
            "facts",
            "top1_choice",
            "params_q16",
            "params_meta",
            "defaults_q16",
            "patches",
            "provenance",
            "constraints",
        ],
    )

    m, k_max, n_max, _iter_n, k, model_ids = _extract_dims(runtime_data)
    _validate_fixed_point(runtime_data.get("fixed_point", {}) or {})

    facts_by_o = _validate_facts(runtime_data.get("facts", {}) or {}, k=k, n_max=n_max)
    fact_counts = _build_fact_counts(facts_by_o, k)

    defaults_norm = _normalize_defaults(runtime_data.get("defaults_q16", {}) or {}, m=m)

    top1_choice = _validate_top1_choice(
        runtime_data.get("top1_choice", {}) or {},
        model_ids=model_ids,
        k=k,
        fact_counts=fact_counts,
    )

    dense_q16 = _init_dense_q16(m=m, k_max=k_max, n_max=n_max, defaults=defaults_norm)
    _apply_patches(
        runtime_data.get("patches", {}) or {},
        dense_q16=dense_q16,
        fact_counts=fact_counts,
        m=m,
        k=k,
    )

    objects_dense = _build_objects_dense(
        objects_raw=runtime_data.get("objects", []) or [],
        facts_by_o=facts_by_o,
        top1_choice=top1_choice,
        model_ids=model_ids,
        m=m,
        k=k,
        k_max=k_max,
        n_max=n_max,
    )

    _validate_optional_constraints(
        dense_q16=dense_q16,
        objects_dense=objects_dense,
        constraints=runtime_data.get("constraints", {}) or {},
        m=m,
        k=k,
        n_max=n_max,
    )

    return {
        "shape": runtime_data.get("shape"),
        "fixed_point": runtime_data.get("fixed_point"),
        "runtime": runtime_data.get("runtime"),
        "objects_dense": objects_dense,
        "params_q16": runtime_data.get("params_q16"),
        "params_meta": runtime_data.get("params_meta"),
        "defaults_q16": {
            "dep_avg": list(defaults_norm["dep_avg"]),
            "support_default": str(defaults_norm["support_default"]),
            "imp_weight_default": str(defaults_norm["imp_weight_default"]),
            "conf_weight_default": str(defaults_norm["conf_weight_default"]),
        },
        "dense_q16": dense_q16,
        "provenance": runtime_data.get("provenance"),
        "constraints": runtime_data.get("constraints"),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Expand truthfinder_runtime_input.json sparse patches into fixed-shape dense arrays "
            "for downstream circom input preparation."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to truthfinder_runtime_input.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write truthfinder_dense_input.json (default: same dir as input)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input
    output_path: Path = args.output or (input_path.parent / "truthfinder_dense_input.json")

    if not input_path.exists():
        raise ExpansionError(f"input file not found: {input_path}")

    runtime_data = _read_json(input_path)
    dense_data = expand_runtime_input(runtime_data)
    _write_json(output_path, dense_data)

    print(f"[expander] wrote dense input: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExpansionError as ex:
        print(f"[expander] error: {ex}")
        raise SystemExit(1)