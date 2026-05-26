from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


Q16_MIN = 0
Q16_MAX = 65536
Q16_SCALE = 1 << 16
EXPECTED_M = 4
EXPECTED_K_MAX = 10
EXPECTED_N_MAX = 8
EXPECTED_ITER_N = 15
FALLBACK_FACT = "(空)"
GENERATOR_VERSION = "build_truthfinder_runtime_input_from_state-v3-top1"
FIXED_PARAMS_Q16: Dict[str, str] = {
    "t0": "49152",
    "beta": "22938",
    "gamma": "19661",
    "alpha_imp": "13107",
    "alpha_conflict": "6554",
    "min_tau_scale": "26214",
    "init_last_s": "32768",
    "trust_prior_default": "49152",
    "trust_prior_strength": "131072",
}

_THIS_FILE = Path(__file__).resolve()
_ZK_DIR = _THIS_FILE.parent
_APP_DIR = _ZK_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent
_DEFAULT_SCHEMA_PATH = _ZK_DIR / "truthfinder_runtime_input_schema.json"
_DEFAULT_TRUTHFINDER_PATH = _APP_DIR / "TruthFinder.py"
_DEFAULT_NORMALIZE_PATH = _APP_DIR / "normalize.py"
_DEFAULT_APP_PATH = _APP_DIR / "app.py"

if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from TruthFinder import (  # type: ignore
    TruthFinderConfig,
    build_cluster_relation_matrix,
    cluster_facts_for_object,
    compute_dependency_with_family,
    compute_rho_dependency_top1,
)
from normalize import normalize_meaning_zh_soft  # type: ignore


class RuntimeInputBuildError(ValueError):
    """Raised when runtime input data is inconsistent with schema constraints."""


class ExpansionError(ValueError):
    """Raised when runtime input cannot be expanded safely."""


def _read_json(path: Path, err_type: type[ValueError] = RuntimeInputBuildError) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as ex:
        raise err_type(f"failed to decode JSON as utf-8: {path}") from ex
    except json.JSONDecodeError as ex:
        raise err_type(f"invalid JSON: {path}: {ex}") from ex


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _require_top_level(data: Mapping[str, Any], required: Sequence[str], err_type: type[ValueError]) -> None:
    missing = [k for k in required if k not in data]
    if missing:
        raise err_type(f"missing required top-level fields: {', '.join(missing)}")


def _as_int(value: Any, field: str, err_type: type[ValueError]) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as ex:
        raise err_type(f"{field} must be int-like, got {value!r}") from ex


def _stable_json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(data: Any) -> str:
    return hashlib.sha256(_stable_json_bytes(data)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_with_fallbacks(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeInputBuildError(f"unable to decode JSON template: {path}")


def _resolve_schema_path(schema_path: str | Path | None) -> Path:
    if schema_path is None:
        return _DEFAULT_SCHEMA_PATH

    p = Path(schema_path)
    if p.is_absolute():
        return p

    cwd_candidate = p.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    zk_candidate = (_ZK_DIR / p).resolve()
    if zk_candidate.exists():
        return zk_candidate

    return zk_candidate


def _resolve_source_path(given: str | Path | None, default_path: Path) -> Path:
    if given is None:
        return default_path.resolve()

    p = Path(given)
    if p.is_absolute():
        return p

    cwd_candidate = p.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    default_dir_candidate = (default_path.parent / p).resolve()
    if default_dir_candidate.exists():
        return default_dir_candidate

    app_candidate = (_APP_DIR / p).resolve()
    if app_candidate.exists():
        return app_candidate

    project_candidate = (_PROJECT_ROOT / p).resolve()
    if project_candidate.exists():
        return project_candidate

    zk_candidate = (_ZK_DIR / p).resolve()
    if zk_candidate.exists():
        return zk_candidate

    return default_dir_candidate


def _float_to_q16_str(value: float, scale: int = Q16_SCALE) -> str:
    q = int((Decimal(str(value)) * Decimal(scale)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    q = max(0, min(scale, q))
    return str(q)


def _signed_float_to_q16_str(value: float, scale: int = Q16_SCALE) -> str:
    q = int((Decimal(str(value)) * Decimal(scale)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    q = max(-scale, min(scale, q))
    return str(q)


def _extract_kw2raw(rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    kw2raw: Dict[str, str] = {}
    for r in rows:
        kw = str(r.get("keyword", "")).strip()
        meaning = str(r.get("meaning_zh", "")).strip()
        if kw:
            kw2raw[kw] = meaning
    return kw2raw


def _sanitize_results_for_hash(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for model in sorted(results.keys()):
        payload = results.get(model, {}) or {}
        keywords_rows: List[Dict[str, str]] = []
        for row in payload.get("keywords", []) or []:
            if not isinstance(row, Mapping):
                continue
            keywords_rows.append(
                {
                    "keyword": str(row.get("keyword", "")),
                    "meaning_zh": str(row.get("meaning_zh", "")),
                }
            )
        out[str(model)] = {
            "translation_zh": str(payload.get("translation_zh", "")),
            "keywords": keywords_rows,
        }
    return out


def _normalize_candidates_for_model(
    model: str,
    keywords: Sequence[str],
    results: Mapping[str, Mapping[str, Any]],
    topn_candidates: int,
    normalized_by_model: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
) -> Dict[str, List[str]]:
    payload = results.get(model, {}) or {}
    kw2raw = _extract_kw2raw(payload.get("keywords", []) or [])
    provided = (normalized_by_model or {}).get(model, {}) or {}

    out: Dict[str, List[str]] = {}
    for kw in keywords:
        if kw in provided:
            cands = [str(x).strip() for x in (provided.get(kw) or []) if str(x).strip()]
        else:
            raw = kw2raw.get(kw, "")
            cands = normalize_meaning_zh_soft(raw, top_n=topn_candidates)
            cands = [c.strip() for c in cands if c and c.strip()]

        dedup: List[str] = []
        seen: Set[str] = set()
        for c in cands:
            if c not in seen:
                seen.add(c)
                dedup.append(c)
        out[kw] = dedup
    return out


def _rank_weight(rank: int, decay: float) -> float:
    decay = min(max(float(decay), 1e-6), 1.0)
    return decay**rank


def _build_support_hint(candidates_by_model: Sequence[Sequence[str]], cfg: TruthFinderConfig) -> Dict[str, float]:
    hint: Dict[str, float] = {}
    for cands in candidates_by_model:
        if not cands:
            continue
        denom = sum(_rank_weight(i, cfg.cand_decay) for i in range(len(cands)))
        denom = denom if denom > 0 else 1.0
        for i, cand in enumerate(cands):
            hint[cand] = hint.get(cand, 0.0) + (_rank_weight(i, cfg.cand_decay) / denom)
    return hint


def _extract_dims_from_runtime_like(
    data: Mapping[str, Any],
    err_type: type[ValueError],
) -> Tuple[int, int, int, int, int, List[str]]:
    shape = data.get("shape", {}) or {}
    runtime = data.get("runtime", {}) or {}

    m = _as_int(shape.get("M"), "shape.M", err_type)
    k_max = _as_int(shape.get("K_MAX"), "shape.K_MAX", err_type)
    n_max = _as_int(shape.get("N_MAX"), "shape.N_MAX", err_type)
    iter_n = _as_int(shape.get("ITER_N"), "shape.ITER_N", err_type)
    k = _as_int(runtime.get("K"), "runtime.K", err_type)
    model_ids = [str(x) for x in (runtime.get("model_ids", []) or [])]

    if m != EXPECTED_M:
        raise err_type(f"shape.M must be {EXPECTED_M}, got {m}")
    if k_max != EXPECTED_K_MAX:
        raise err_type(f"shape.K_MAX must be {EXPECTED_K_MAX}, got {k_max}")
    if n_max != EXPECTED_N_MAX:
        raise err_type(f"shape.N_MAX must be {EXPECTED_N_MAX}, got {n_max}")
    if iter_n != EXPECTED_ITER_N:
        raise err_type(f"shape.ITER_N must be {EXPECTED_ITER_N}, got {iter_n}")
    if k < 0 or k > k_max:
        raise err_type(f"runtime.K must be in [0, {k_max}], got {k}")
    if len(model_ids) != m:
        raise err_type(f"runtime.model_ids length {len(model_ids)} must equal M={m}")

    return m, k_max, n_max, iter_n, k, model_ids


def _validate_runtime_input(data: Mapping[str, Any]) -> None:
    m, _k_max, n_max, _iter_n, k, model_ids = _extract_dims_from_runtime_like(data, RuntimeInputBuildError)

    runtime = data.get("runtime", {}) or {}
    if str(runtime.get("support_mode", "")) != "top1_in_circuit":
        raise RuntimeInputBuildError(
            f"runtime.support_mode must be 'top1_in_circuit', got {runtime.get('support_mode')!r}"
        )

    if dict(data.get("params_q16", {}) or {}) != FIXED_PARAMS_Q16:
        raise RuntimeInputBuildError("params_q16 must exactly match fixed v2 circuit parameters")

    facts_raw = data.get("facts", {}) or {}
    if not isinstance(facts_raw, Mapping):
        raise RuntimeInputBuildError("facts must be an object")

    facts: Dict[int, List[str]] = {}
    for o in range(k):
        key = str(o)
        if key not in facts_raw:
            raise RuntimeInputBuildError(f"facts missing effective object entry {key}")
        arr = facts_raw[key]
        if not isinstance(arr, list):
            raise RuntimeInputBuildError(f"facts[{key}] must be a list")
        if len(arr) == 0 or len(arr) > n_max:
            raise RuntimeInputBuildError(f"facts[{key}] length must be in [1,{n_max}], got {len(arr)}")
        facts[o] = [str(x) for x in arr]

    top1_choice = data.get("top1_choice", {}) or {}
    if set(top1_choice.keys()) != set(model_ids):
        raise RuntimeInputBuildError("top1_choice must contain exactly one object map for each model_id")
    for model in model_ids:
        obj_map = top1_choice.get(model, {}) or {}
        if not isinstance(obj_map, Mapping):
            raise RuntimeInputBuildError(f"top1_choice[{model!r}] must be an object map")
        for o in range(k):
            key = str(o)
            if key not in obj_map:
                raise RuntimeInputBuildError(f"top1_choice[{model}] missing effective object {key}")
            idx = _as_int(obj_map[key], f"top1_choice[{model}][{key}]", RuntimeInputBuildError)
            n_o = len(facts[o])
            if idx < 0 or idx >= n_o:
                raise RuntimeInputBuildError(
                    f"top1_choice[{model}][{key}] must be in [0,{n_o - 1}] for effective object, got {idx}"
                )

    patches = data.get("patches", {}) or {}
    if "support_patch" in patches:
        raise RuntimeInputBuildError("patches.support_patch is not allowed in v2 top1_in_circuit runtime input")

    dep_patch = patches.get("dep_avg_patch", []) or []
    relation_patch = patches.get("relation_patch", []) or []
    imp_patch = patches.get("imp_weight_patch", []) or []
    conf_patch = patches.get("conf_weight_patch", []) or []

    if len(dep_patch) != m:
        raise RuntimeInputBuildError(f"patches.dep_avg_patch must contain exactly M={m} entries")

    dep_seen: Set[int] = set()
    for i, patch in enumerate(dep_patch):
        if not isinstance(patch, Mapping):
            raise RuntimeInputBuildError(f"patches.dep_avg_patch[{i}] must be an object")
        w = _as_int(patch.get("w"), f"patches.dep_avg_patch[{i}].w", RuntimeInputBuildError)
        v = _as_int(patch.get("value"), f"patches.dep_avg_patch[{i}].value", RuntimeInputBuildError)
        if w < 0 or w >= m:
            raise RuntimeInputBuildError(f"patches.dep_avg_patch[{i}].w out of range [0,{m - 1}]")
        if v < 0 or v > Q16_MAX:
            raise RuntimeInputBuildError(f"patches.dep_avg_patch[{i}].value out of range [0,{Q16_MAX}]")
        dep_seen.add(w)
    if dep_seen != set(range(m)):
        raise RuntimeInputBuildError("patches.dep_avg_patch must cover every model index exactly once")

    imp_map: Dict[Tuple[int, int, int], int] = {}
    conf_map: Dict[Tuple[int, int, int], int] = {}

    for name, arr, lo, hi, target in (
        ("relation_patch", relation_patch, -Q16_MAX, Q16_MAX, None),
        ("imp_weight_patch", imp_patch, 0, Q16_MAX, imp_map),
        ("conf_weight_patch", conf_patch, 0, Q16_MAX, conf_map),
    ):
        for i, patch in enumerate(arr):
            if not isinstance(patch, Mapping):
                raise RuntimeInputBuildError(f"patches.{name}[{i}] must be an object")
            o = _as_int(patch.get("o"), f"patches.{name}[{i}].o", RuntimeInputBuildError)
            g = _as_int(patch.get("g"), f"patches.{name}[{i}].g", RuntimeInputBuildError)
            f = _as_int(patch.get("f"), f"patches.{name}[{i}].f", RuntimeInputBuildError)
            v = _as_int(patch.get("value"), f"patches.{name}[{i}].value", RuntimeInputBuildError)
            if o < 0 or o >= k:
                raise RuntimeInputBuildError(f"patches.{name}[{i}].o out of range [0,{k - 1}]")
            n_o = len(facts[o])
            if g < 0 or g >= n_o or f < 0 or f >= n_o:
                raise RuntimeInputBuildError(
                    f"patches.{name}[{i}] has invalid indices for object {o} with N_o={n_o}: g={g}, f={f}"
                )
            if v < lo or v > hi:
                raise RuntimeInputBuildError(f"patches.{name}[{i}].value out of range [{lo},{hi}]")
            if g == f and v != 0:
                raise RuntimeInputBuildError(f"patches.{name}[{i}] diagonal entries must be 0")
            if target is not None:
                target[(o, g, f)] = v

    for key in set(imp_map) & set(conf_map):
        if imp_map[key] != 0 and conf_map[key] != 0:
            raise RuntimeInputBuildError(f"imp/conf patches must be exclusive at edge {key}")

    fact_counts = [len(facts[o]) for o in range(k)]
    try:
        _validate_relation_consistency(patches, fact_counts=fact_counts, k=k)
    except ExpansionError as ex:
        raise RuntimeInputBuildError(str(ex)) from ex


def _normalize_defaults(defaults_q16: Mapping[str, Any], m: int) -> Dict[str, Any]:
    required = ["dep_avg", "imp_weight_default", "conf_weight_default"]
    missing = [k for k in required if k not in defaults_q16]
    if missing:
        raise ExpansionError(f"defaults_q16 missing required fields: {', '.join(missing)}")

    dep_raw = defaults_q16.get("dep_avg")
    if not isinstance(dep_raw, list) or len(dep_raw) != m:
        raise ExpansionError(f"defaults_q16.dep_avg must be a list of length M={m}")
    dep_avg = [_parse_dense_q16(dep_raw[w], f"defaults_q16.dep_avg[{w}]") for w in range(m)]

    imp_default = _parse_dense_q16(defaults_q16.get("imp_weight_default"), "defaults_q16.imp_weight_default")
    conf_default = _parse_dense_q16(defaults_q16.get("conf_weight_default"), "defaults_q16.conf_weight_default")
    relation_default = defaults_q16.get("relation_default")
    top1_default = defaults_q16.get("top1_default", -1)

    return {
        "dep_avg": dep_avg,
        "imp_weight_default": imp_default,
        "conf_weight_default": conf_default,
        "relation_default": None if relation_default is None else str(relation_default),
        "top1_default": int(top1_default),
    }


def _parse_dense_q16(value: Any, field: str) -> str:
    i = _as_int(value, field, ExpansionError)
    if i < Q16_MIN or i > Q16_MAX:
        raise ExpansionError(f"{field} out of Q16 range [{Q16_MIN}, {Q16_MAX}]: {i}")
    return str(i)


def _validate_facts_for_expand(facts: Mapping[str, Any], k: int, n_max: int) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for o in range(k):
        key = str(o)
        if key not in facts:
            raise ExpansionError(f"facts missing effective object entry {key}")
        arr = facts[key]
        if not isinstance(arr, list):
            raise ExpansionError(f"facts[{key}] must be a list")
        if len(arr) == 0 or len(arr) > n_max:
            raise ExpansionError(f"facts[{key}] length must be in [1,{n_max}], got {len(arr)}")
        out[o] = [str(x) for x in arr]
    return out


def _validate_top1_choice_for_expand(
    top1_choice: Mapping[str, Any],
    model_ids: Sequence[str],
    k: int,
    fact_counts: Sequence[int],
) -> Dict[str, Dict[int, int]]:
    out: Dict[str, Dict[int, int]] = {m: {} for m in model_ids}
    if set(top1_choice.keys()) != set(model_ids):
        raise ExpansionError("top1_choice must contain exactly one object map for each model_id")

    for model in model_ids:
        obj_map = top1_choice.get(model, {}) or {}
        if not isinstance(obj_map, Mapping):
            raise ExpansionError(f"top1_choice[{model!r}] must be an object map")
        parsed: Dict[int, int] = {}
        for o in range(k):
            key = str(o)
            if key not in obj_map:
                raise ExpansionError(f"top1_choice[{model}] missing effective object {key}")
            idx = _as_int(obj_map[key], f"top1_choice[{model}][{key}]", ExpansionError)
            n_o = fact_counts[o]
            if idx < 0 or idx >= n_o:
                raise ExpansionError(
                    f"top1_choice[{model}][{key}] must be in [0,{n_o - 1}] for effective object, got {idx}"
                )
            parsed[o] = idx
        out[model] = parsed
    return out


def _build_fact_counts(facts_by_o: Mapping[int, Sequence[str]], k: int) -> List[int]:
    return [len(facts_by_o.get(o, [])) for o in range(k)]


def _init_dense_q16(m: int, k_max: int, n_max: int, defaults: Mapping[str, Any]) -> Dict[str, Any]:
    imp_default = str(defaults["imp_weight_default"])
    conf_default = str(defaults["conf_weight_default"])
    return {
        "dep_avg": list(defaults["dep_avg"]),
        "imp_weight": [
            [[imp_default for _f in range(n_max)] for _g in range(n_max)]
            for _o in range(k_max)
        ],
        "conf_weight": [
            [[conf_default for _f in range(n_max)] for _g in range(n_max)]
            for _o in range(k_max)
        ],
    }


def _validate_relation_consistency(
    patches: Mapping[str, Any],
    fact_counts: Sequence[int],
    k: int,
) -> None:
    relation_patch = patches.get("relation_patch", []) or []
    imp_patch = patches.get("imp_weight_patch", []) or []
    conf_patch = patches.get("conf_weight_patch", []) or []

    relation_map: Dict[Tuple[int, int, int], int] = {}
    imp_map: Dict[Tuple[int, int, int], int] = {}
    conf_map: Dict[Tuple[int, int, int], int] = {}

    for name, arr, lo, hi, target in (
        ("relation_patch", relation_patch, -Q16_MAX, Q16_MAX, relation_map),
        ("imp_weight_patch", imp_patch, 0, Q16_MAX, imp_map),
        ("conf_weight_patch", conf_patch, 0, Q16_MAX, conf_map),
    ):
        for i, patch in enumerate(arr):
            if not isinstance(patch, Mapping):
                raise ExpansionError(f"patches.{name}[{i}] must be an object")
            o = _as_int(patch.get("o"), f"patches.{name}[{i}].o", ExpansionError)
            g = _as_int(patch.get("g"), f"patches.{name}[{i}].g", ExpansionError)
            f = _as_int(patch.get("f"), f"patches.{name}[{i}].f", ExpansionError)
            v = _as_int(patch.get("value"), f"patches.{name}[{i}].value", ExpansionError)
            if o < 0 or o >= k:
                raise ExpansionError(f"patches.{name}[{i}].o out of range [0,{k - 1}]")
            n_o = fact_counts[o]
            if g < 0 or g >= n_o or f < 0 or f >= n_o:
                raise ExpansionError(
                    f"patches.{name}[{i}] has invalid indices for object {o} with N_o={n_o}: g={g}, f={f}"
                )
            if v < lo or v > hi:
                raise ExpansionError(f"patches.{name}[{i}].value out of range [{lo},{hi}]")
            if g == f and v != 0:
                raise ExpansionError(f"patches.{name}[{i}] diagonal entries must be 0")
            target[(o, g, f)] = v

    for key in set(imp_map) & set(conf_map):
        if imp_map[key] != 0 and conf_map[key] != 0:
            raise ExpansionError(f"imp/conf patches must be exclusive at edge {key}")

    for key, rel in relation_map.items():
        imp_val = imp_map.get(key, 0)
        conf_val = conf_map.get(key, 0)
        if rel > 0 and imp_val != rel:
            raise ExpansionError(f"relation/imp mismatch at edge {key}: relation={rel}, imp={imp_val}")
        if rel > 0 and conf_val != 0:
            raise ExpansionError(f"positive relation cannot also have conf patch at edge {key}")
        if rel < 0 and conf_val != -rel:
            raise ExpansionError(f"relation/conf mismatch at edge {key}: relation={rel}, conf={conf_val}")
        if rel < 0 and imp_val != 0:
            raise ExpansionError(f"negative relation cannot also have imp patch at edge {key}")
        if rel == 0 and (imp_val != 0 or conf_val != 0):
            raise ExpansionError(f"zero relation cannot have imp/conf patches at edge {key}")

    for key, imp_val in imp_map.items():
        if imp_val != 0 and relation_map.get(key) != imp_val:
            raise ExpansionError(
                f"imp patch requires matching positive relation at edge {key}: imp={imp_val}, relation={relation_map.get(key)}"
            )
    for key, conf_val in conf_map.items():
        if conf_val != 0 and relation_map.get(key) != -conf_val:
            raise ExpansionError(
                f"conf patch requires matching negative relation at edge {key}: conf={conf_val}, relation={relation_map.get(key)}"
            )


def _apply_dense_patches(
    patches: Mapping[str, Any],
    dense_q16: MutableMapping[str, Any],
    fact_counts: Sequence[int],
    m: int,
    k: int,
) -> None:
    dep_patch = patches.get("dep_avg_patch", []) or []
    imp_patch = patches.get("imp_weight_patch", []) or []
    conf_patch = patches.get("conf_weight_patch", []) or []

    for i, patch in enumerate(dep_patch):
        w = _as_int(patch.get("w"), f"patches.dep_avg_patch[{i}].w", ExpansionError)
        if w < 0 or w >= m:
            raise ExpansionError(f"patches.dep_avg_patch[{i}].w out of range [0,{m - 1}]")
        dense_q16["dep_avg"][w] = _parse_dense_q16(patch.get("value"), f"patches.dep_avg_patch[{i}].value")

    seen_imp: Set[Tuple[int, int, int]] = set()
    seen_conf: Set[Tuple[int, int, int]] = set()
    for name, arr, target, seen in (
        ("imp_weight_patch", imp_patch, "imp_weight", seen_imp),
        ("conf_weight_patch", conf_patch, "conf_weight", seen_conf),
    ):
        for i, patch in enumerate(arr):
            o = _as_int(patch.get("o"), f"patches.{name}[{i}].o", ExpansionError)
            g = _as_int(patch.get("g"), f"patches.{name}[{i}].g", ExpansionError)
            f = _as_int(patch.get("f"), f"patches.{name}[{i}].f", ExpansionError)
            if o < 0 or o >= k:
                raise ExpansionError(f"patches.{name}[{i}].o out of range [0,{k - 1}]")
            n_o = fact_counts[o]
            if g < 0 or g >= n_o or f < 0 or f >= n_o:
                raise ExpansionError(
                    f"patches.{name}[{i}] has invalid indices for object {o} with N_o={n_o}: g={g}, f={f}"
                )
            if g == f:
                raise ExpansionError(f"patches.{name}[{i}] diagonal entries are not allowed")
            val = _parse_dense_q16(patch.get("value"), f"patches.{name}[{i}].value")
            key = (o, g, f)
            if key in seen:
                raise ExpansionError(f"duplicate patch for {name} at edge {key}")
            seen.add(key)
            dense_q16[target][o][g][f] = val

    for key in seen_imp & seen_conf:
        if dense_q16["imp_weight"][key[0]][key[1]][key[2]] != "0" and dense_q16["conf_weight"][key[0]][key[1]][key[2]] != "0":
            raise ExpansionError(f"imp/conf patches must be exclusive at edge {key}")


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
        o = _as_int(obj.get("o"), f"objects[{i}].o", ExpansionError)
        if o < 0 or o >= k:
            raise ExpansionError(f"objects[{i}].o={o} out of effective object range [0,{k - 1}]")
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
                    "top1_by_model": [-1 for _ in range(m)],
                }
            )
            continue

        if o not in objects_map:
            raise ExpansionError(f"objects missing effective object entry for o={o}")

        obj_raw = objects_map[o]
        keyword = str(obj_raw.get("keyword", ""))
        facts = list(facts_by_o.get(o, []))
        fact_count = len(facts)
        facts_padded = facts + ["" for _ in range(n_max - fact_count)]

        top1_by_model = [int(top1_choice[model][o]) for model in model_ids]
        dense.append(
            {
                "o": o,
                "keyword": keyword,
                "is_effective": True,
                "fact_count": fact_count,
                "facts_padded": facts_padded,
                "top1_by_model": top1_by_model,
            }
        )

    return dense


def _validate_dense_relation_state(dense_q16: Mapping[str, Any], k: int, fact_counts: Sequence[int]) -> None:
    for o in range(k):
        n_o = fact_counts[o]
        for g in range(n_o):
            for f in range(n_o):
                imp = int(dense_q16["imp_weight"][o][g][f])
                conf = int(dense_q16["conf_weight"][o][g][f])
                if g == f and (imp != 0 or conf != 0):
                    raise ExpansionError(f"imp/conf diagonal must be zero at object={o}, index={g}")
                if imp != 0 and conf != 0:
                    raise ExpansionError(f"imp/conf must be exclusive at object={o}, g={g}, f={f}")


def build_truthfinder_runtime_input_from_state(
    *,
    input_text: str,
    sentence_id: str,
    session_id: str,
    keywords: Sequence[str],
    results: Mapping[str, Mapping[str, Any]],
    cfg: TruthFinderConfig,
    schema_path: str | Path | None = None,
    normalized_by_model: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
    model_ids: Optional[Sequence[str]] = None,
    truthfinder_path: str | Path | None = None,
    normalize_path: str | Path | None = None,
    app_path: str | Path | None = None,
) -> Dict[str, Any]:
    schema_file = _resolve_schema_path(schema_path)
    if not schema_file.exists():
        raise FileNotFoundError(f"schema template not found: {schema_file}")

    template = _load_json_with_fallbacks(schema_file)
    data: Dict[str, Any] = copy.deepcopy(template)

    kws = [str(k).strip() for k in keywords if str(k).strip()]
    if len(kws) > EXPECTED_K_MAX:
        raise RuntimeInputBuildError(f"ZK circuit supports at most {EXPECTED_K_MAX} keywords.")

    if model_ids is not None:
        models = list(model_ids)
    else:
        schema_models = [str(x) for x in ((template.get("runtime", {}) or {}).get("model_ids", []) or [])]
        models = schema_models if schema_models else list(results.keys())
    if len(models) != EXPECTED_M:
        raise RuntimeInputBuildError(f"model_ids length must equal fixed M={EXPECTED_M}, got {len(models)}")
    for model in models:
        if model not in results:
            raise RuntimeInputBuildError(f"model_id {model!r} missing from results")

    runtime = data.setdefault("runtime", {})
    runtime["session_id"] = session_id
    runtime["sentence_id"] = sentence_id
    runtime["input_text"] = input_text
    runtime["K"] = len(kws)
    runtime["model_ids"] = list(models)
    runtime["support_mode"] = "top1_in_circuit"
    runtime["support_construction"] = "support is constructed inside the circuit from top1_choice"

    norm_cands: Dict[str, Dict[str, List[str]]] = {}
    for m in models:
        norm_cands[m] = _normalize_candidates_for_model(
            model=m,
            keywords=kws,
            results=results,
            topn_candidates=cfg.topn_candidates,
            normalized_by_model=normalized_by_model,
        )

    facts_by_o: Dict[str, List[str]] = {}
    cluster_members_by_o: Dict[str, Dict[str, List[str]]] = {}
    top1_choice_idx: Dict[str, Dict[str, int]] = {m: {} for m in models}
    top1_choice_text: Dict[str, Dict[Tuple[str, str], str]] = {m: {} for m in models}
    relation_patch: List[Dict[str, Any]] = []
    imp_patch: List[Dict[str, Any]] = []
    conf_patch: List[Dict[str, Any]] = []

    empty_fact = str(getattr(cfg, "empty_fact", FALLBACK_FACT) or FALLBACK_FACT)

    for o, kw in enumerate(kws):
        candidates_by_model: List[List[str]] = []
        raw_facts: List[str] = []
        seen: Set[str] = set()

        for m in models:
            cands = list(norm_cands[m].get(kw, []))
            if not cands:
                cands = [empty_fact]
            candidates_by_model.append(cands)
            for cand in cands:
                if cand not in seen:
                    seen.add(cand)
                    raw_facts.append(cand)

        if not raw_facts:
            raw_facts = [empty_fact]
            candidates_by_model = [[empty_fact] for _ in models]

        support_hint = _build_support_hint(candidates_by_model, cfg)
        cluster_facts, fact_to_cluster, cluster_members = cluster_facts_for_object(
            raw_facts,
            cfg,
            support_hint=support_hint,
        )
        if not cluster_facts:
            cluster_facts = [empty_fact]
            fact_to_cluster = {empty_fact: empty_fact}
            cluster_members = {empty_fact: [empty_fact]}

        if len(cluster_facts) > EXPECTED_N_MAX:
            raise RuntimeInputBuildError(
                f"object {o} produced {len(cluster_facts)} clustered facts > N_MAX={EXPECTED_N_MAX}"
            )

        facts_by_o[str(o)] = list(cluster_facts)
        cluster_members_by_o[str(o)] = {rep: list(members) for rep, members in cluster_members.items()}
        rep_to_idx = {rep: idx for idx, rep in enumerate(cluster_facts)}

        for w, m in enumerate(models):
            raw_top1 = candidates_by_model[w][0] if candidates_by_model[w] else empty_fact
            rep = fact_to_cluster.get(raw_top1, raw_top1)
            if rep not in rep_to_idx:
                raise RuntimeInputBuildError(
                    f"top1 cluster representative missing from facts: object={o}, model={m}, rep={rep}"
                )
            top1_choice_idx[m][str(o)] = rep_to_idx[rep]
            top1_choice_text[m][(sentence_id, kw)] = rep

        rel = build_cluster_relation_matrix(cluster_facts, cluster_members, cfg)
        for g_idx, g in enumerate(cluster_facts):
            for f_idx, f in enumerate(cluster_facts):
                if g_idx == f_idx:
                    continue
                score = float(rel.get((g, f), 0.0))
                if score == 0.0:
                    continue
                rel_q16 = int(_signed_float_to_q16_str(score))
                if rel_q16 == 0:
                    continue
                relation_patch.append({"o": o, "g": g_idx, "f": f_idx, "value": str(rel_q16)})
                if rel_q16 > 0:
                    imp_patch.append({"o": o, "g": g_idx, "f": f_idx, "value": str(rel_q16)})
                else:
                    conf_patch.append({"o": o, "g": g_idx, "f": f_idx, "value": str(-rel_q16)})

    observed_rho = compute_rho_dependency_top1(models, top1_choice_text)
    dependency = compute_dependency_with_family(models, observed_rho, cfg)
    dep_avg_patch: List[Dict[str, Any]] = []
    for w, model in enumerate(models):
        vals = [float(dependency.get((model, other), 0.0)) for other in models if other != model]
        dep_avg = (sum(vals) / len(vals)) if vals else 0.0
        dep_avg_patch.append({"w": w, "value": _float_to_q16_str(dep_avg)})

    data["objects"] = [{"o": o, "keyword": kw, "is_effective": True} for o, kw in enumerate(kws)]
    data["facts"] = facts_by_o
    data["cluster_members"] = cluster_members_by_o
    data["top1_choice"] = top1_choice_idx
    data["params_q16"] = dict(FIXED_PARAMS_Q16)

    params_meta = data.setdefault("params_meta", {})
    params_meta.update(
        {
            "topn_candidates": int(getattr(cfg, "topn_candidates", 3)),
            "cand_decay": float(getattr(cfg, "cand_decay", 0.30)),
            "merge_literal_threshold": float(getattr(cfg, "merge_literal_threshold", 0.82)),
            "merge_containment": bool(getattr(cfg, "merge_containment", False)),
            "rel_exact_score": float(getattr(cfg, "rel_exact_score", 0.90)),
            "rel_conflict_score": float(getattr(cfg, "rel_conflict_score", 0.70)),
            "rel_synonym_score": float(getattr(cfg, "rel_synonym_score", 0.70)),
            "rel_containment_score": float(getattr(cfg, "rel_containment_score", 0.40)),
            "rel_literal_scale": float(getattr(cfg, "rel_literal_scale", 0.60)),
            "imp_sim_threshold": float(getattr(cfg, "imp_sim_threshold", 0.45)),
            "max_iter": int(getattr(cfg, "max_iter", EXPECTED_ITER_N)),
            "python_default_early_stop": bool(getattr(cfg, "early_stop", True)),
            "zk_iter_mode": "fixed_15_rounds",
            "min_iter": int(getattr(cfg, "min_iter", 2)),
            "delta": float(getattr(cfg, "delta", 0.0001)),
            "abs_delta": float(getattr(cfg, "abs_delta", 0.0001)),
            "support_mode": "top1_in_circuit",
            "use_family_dependency": bool(getattr(cfg, "use_family_dependency", True)),
            "family_dep_same": float(getattr(cfg, "family_dep_same", 0.50)),
            "family_dep_unknown": float(getattr(cfg, "family_dep_unknown", 0.10)),
            "family_dep_different": float(getattr(cfg, "family_dep_different", 0.0)),
            "use_trust_prior": True,
            "trust_prior_default": 0.75,
            "trust_prior_strength": 2.0,
        }
    )

    facts_meta = data.setdefault("facts_meta", {})
    facts_meta["max_candidates_per_object"] = EXPECTED_N_MAX
    facts_meta["empty_fact"] = empty_fact

    data["patches"] = {
        "dep_avg_patch": dep_avg_patch,
        "relation_patch": relation_patch,
        "imp_weight_patch": imp_patch,
        "conf_weight_patch": conf_patch,
    }

    truthfinder_file = _resolve_source_path(truthfinder_path, _DEFAULT_TRUTHFINDER_PATH)
    normalize_file = _resolve_source_path(normalize_path, _DEFAULT_NORMALIZE_PATH)
    app_file = _resolve_source_path(app_path, _DEFAULT_APP_PATH)
    for fp in (truthfinder_file, normalize_file, app_file):
        if not fp.exists():
            raise FileNotFoundError(f"required source file not found: {fp}")

    provenance = data.setdefault("provenance", {})
    provenance["truthfinder_version"] = f"TruthFinder.py@{_file_sha256(truthfinder_file)[:12]}"
    provenance["normalize_version"] = f"normalize.py@{_file_sha256(normalize_file)[:12]}"
    provenance["app_version"] = f"app.py@{_file_sha256(app_file)[:12]}"
    provenance["generator_version"] = GENERATOR_VERSION

    support_rule = "top1_in_circuit: support[o][top1_choice[o,w]][w]=65536"
    provenance["facts_hash"] = _sha256_hex({"facts": data["facts"]})
    provenance["cluster_hash"] = _sha256_hex({"cluster_members": data["cluster_members"]})
    provenance["relation_hash"] = _sha256_hex({"relation_patch": relation_patch})
    provenance["imp_hash"] = _sha256_hex({"imp_weight_patch": imp_patch})
    provenance["conf_hash"] = _sha256_hex({"conf_weight_patch": conf_patch})
    provenance["top1_hash"] = _sha256_hex({"top1_choice": data["top1_choice"]})
    provenance["dep_hash"] = _sha256_hex({"dep_avg_patch": dep_avg_patch})
    provenance["params_hash"] = _sha256_hex({"params_q16": data["params_q16"]})
    provenance["support_rule_hash"] = _sha256_hex({"support_rule": support_rule})
    provenance["normalize_hash"] = _sha256_hex({"normalized_candidates": norm_cands})
    provenance["input_hash"] = _sha256_hex(
        {
            "session_id": session_id,
            "sentence_id": sentence_id,
            "input_text": input_text,
            "keywords": kws,
            "results": _sanitize_results_for_hash(results),
        }
    )
    provenance.pop("support_hash", None)

    _validate_runtime_input(data)
    return data


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
        ExpansionError,
    )
    try:
        _validate_runtime_input(runtime_data)
    except RuntimeInputBuildError as ex:
        raise ExpansionError(str(ex)) from ex

    m, k_max, n_max, _iter_n, k, model_ids = _extract_dims_from_runtime_like(runtime_data, ExpansionError)
    runtime = runtime_data.get("runtime", {}) or {}
    if str(runtime.get("support_mode", "")) != "top1_in_circuit":
        raise ExpansionError(
            f"runtime.support_mode must be 'top1_in_circuit', got {runtime.get('support_mode')!r}"
        )

    if dict(runtime_data.get("params_q16", {}) or {}) != FIXED_PARAMS_Q16:
        raise ExpansionError("params_q16 must exactly match fixed v2 circuit parameters")

    facts_by_o = _validate_facts_for_expand(runtime_data.get("facts", {}) or {}, k=k, n_max=n_max)
    fact_counts = _build_fact_counts(facts_by_o, k)
    defaults_norm = _normalize_defaults(runtime_data.get("defaults_q16", {}) or {}, m=m)
    top1_choice = _validate_top1_choice_for_expand(
        runtime_data.get("top1_choice", {}) or {},
        model_ids=model_ids,
        k=k,
        fact_counts=fact_counts,
    )

    _validate_relation_consistency(runtime_data.get("patches", {}) or {}, fact_counts=fact_counts, k=k)

    dense_q16 = _init_dense_q16(m=m, k_max=k_max, n_max=n_max, defaults=defaults_norm)
    _apply_dense_patches(
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

    _validate_dense_relation_state(dense_q16, k=k, fact_counts=fact_counts)

    defaults_out: Dict[str, Any] = {
        "dep_avg": list(defaults_norm["dep_avg"]),
        "imp_weight_default": str(defaults_norm["imp_weight_default"]),
        "conf_weight_default": str(defaults_norm["conf_weight_default"]),
    }
    if defaults_norm["relation_default"] is not None:
        defaults_out["relation_default"] = defaults_norm["relation_default"]
    defaults_out["top1_default"] = str(defaults_norm["top1_default"])

    return {
        "shape": runtime_data.get("shape"),
        "fixed_point": runtime_data.get("fixed_point"),
        "runtime": runtime_data.get("runtime"),
        "objects_dense": objects_dense,
        "params_q16": runtime_data.get("params_q16"),
        "params_meta": runtime_data.get("params_meta"),
        "defaults_q16": defaults_out,
        "dense_q16": dense_q16,
        "provenance": runtime_data.get("provenance"),
        "constraints": runtime_data.get("constraints"),
    }


def build_dense_input_from_state(**kwargs: Any) -> Dict[str, Any]:
    runtime_data = build_truthfinder_runtime_input_from_state(**kwargs)
    return expand_runtime_input(runtime_data)


def save_runtime_input_json(path: str | Path, data: Mapping[str, Any]) -> None:
    _write_json(Path(path), data)


def save_dense_input_json(path: str | Path, data: Mapping[str, Any]) -> None:
    _write_json(Path(path), data)


def _build_expander_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Expand truthfinder_runtime_input.json into truthfinder_dense_input.json "
            "for the v2 top1-in-circuit ZK pipeline."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to truthfinder_runtime_input.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write truthfinder_dense_input.json (default: same dir as input)",
    )
    return parser


def expander_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_expander_arg_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input
    output_path: Path = args.output or (input_path.parent / "truthfinder_dense_input.json")
    if not input_path.exists():
        raise ExpansionError(f"input file not found: {input_path}")

    runtime_data = _read_json(input_path, ExpansionError)
    dense_data = expand_runtime_input(runtime_data)
    save_dense_input_json(output_path, dense_data)
    print(f"[expander] wrote dense input: {output_path}")
    return 0


__all__ = [
    "RuntimeInputBuildError",
    "ExpansionError",
    "build_truthfinder_runtime_input_from_state",
    "expand_runtime_input",
    "build_dense_input_from_state",
    "save_runtime_input_json",
    "save_dense_input_json",
    "expander_main",
]
