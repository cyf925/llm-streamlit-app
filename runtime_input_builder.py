from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from TruthFinder import (
    TruthFinderConfig,
    _candidate_weights,
    build_conflict_matrix,
    build_implication_matrix,
    compute_rho_dependency_top1,
)
from normalize import normalize_meaning_zh_soft


Q16_SCALE = 1 << 16
FALLBACK_FACT = "(空)"
GENERATOR_VERSION = "build_truthfinder_runtime_input_from_state-v1"


class RuntimeInputBuildError(ValueError):
    """Raised when runtime input data is inconsistent with schema constraints."""


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


def _float_to_q16_str(value: float, scale: int = Q16_SCALE) -> str:
    # Deterministic half-up rounding, then clamp into [0, scale].
    q = int((Decimal(str(value)) * Decimal(scale)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    q = max(0, min(scale, q))
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
    # Keep only stable, JSON-serializable fields to avoid hash drift from non-serializable runtime objects.
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


def _coerce_patch_indices_to_int(data: Dict[str, Any]) -> None:
    patches = data.get("patches", {}) or {}

    for item in patches.get("dep_avg_patch", []) or []:
        item["w"] = int(item["w"])

    for item in patches.get("support_patch", []) or []:
        item["o"] = int(item["o"])
        item["f"] = int(item["f"])
        item["w"] = int(item["w"])

    for key in ("imp_weight_patch", "conf_weight_patch"):
        for item in patches.get(key, []) or []:
            item["o"] = int(item["o"])
            item["g"] = int(item["g"])
            item["f"] = int(item["f"])


def _normalize_candidates_for_model(
    model: str,
    keywords: Sequence[str],
    results: Mapping[str, Mapping[str, Any]],
    topn_candidates: int,
    normalized_by_model: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
) -> Dict[str, List[str]]:
    payload = results.get(model, {}) or {}
    kw2raw = _extract_kw2raw(payload.get("keywords", []) or [])

    out: Dict[str, List[str]] = {}
    provided = (normalized_by_model or {}).get(model, {}) or {}

    for kw in keywords:
        if kw in provided:
            cands = [str(x).strip() for x in (provided.get(kw) or []) if str(x).strip()]
        else:
            raw = kw2raw.get(kw, "")
            cands = normalize_meaning_zh_soft(raw, top_n=topn_candidates)
            cands = [c.strip() for c in cands if c and c.strip()]

        # stable de-dup in-order
        dedup: List[str] = []
        seen = set()
        for c in cands:
            if c not in seen:
                seen.add(c)
                dedup.append(c)
        out[kw] = dedup
    return out


def _validate_runtime_input(data: Mapping[str, Any]) -> None:
    shape = data.get("shape", {})
    m_max = int(shape.get("M", 0))
    k_max = int(shape.get("K_MAX", 0))
    n_max = int(shape.get("N_MAX", 0))

    runtime = data.get("runtime", {})
    model_ids: List[str] = runtime.get("model_ids", []) or []
    k = int(runtime.get("K", 0))

    if len(model_ids) > m_max:
        raise RuntimeInputBuildError(f"model count {len(model_ids)} exceeds M={m_max}")
    if k > k_max:
        raise RuntimeInputBuildError(f"K={k} exceeds K_MAX={k_max}")

    facts: Dict[str, List[str]] = data.get("facts", {}) or {}
    for o_str, fact_list in facts.items():
        if len(fact_list) > n_max:
            raise RuntimeInputBuildError(f"object {o_str} has {len(fact_list)} facts > N_MAX={n_max}")

    # top1 in range [-1, N_o-1]
    top1_choice = data.get("top1_choice", {}) or {}
    for model, obj_map in top1_choice.items():
        for o_str, idx in (obj_map or {}).items():
            o_int = int(o_str)
            n_o = len(facts.get(str(o_int), []))
            if not (-1 <= int(idx) <= n_o - 1):
                raise RuntimeInputBuildError(
                    f"top1_choice out of range: model={model}, o={o_int}, idx={idx}, N_o={n_o}"
                )

    # support row sum check: for each (o,w) row with any support -> approx 65536
    support_rows: Dict[Tuple[int, int], int] = {}
    model_to_idx = {m: i for i, m in enumerate(model_ids)}
    for patch in data.get("patches", {}).get("support_patch", []) or []:
        o, w = int(patch["o"]), int(patch["w"])
        if o >= k or w >= len(model_ids):
            raise RuntimeInputBuildError(f"support_patch index out of bounds: {patch}")
        support_rows[(o, w)] = support_rows.get((o, w), 0) + int(patch["value"])

    for (o, w), row_sum in support_rows.items():
        if abs(row_sum - Q16_SCALE) > 1:
            model = model_ids[w]
            raise RuntimeInputBuildError(f"support row sum invalid at (o={o}, model={model}): {row_sum}")

    imp_patch = data.get("patches", {}).get("imp_weight_patch", []) or []
    conf_patch = data.get("patches", {}).get("conf_weight_patch", []) or []

    for patch in imp_patch:
        o, g, f = int(patch["o"]), int(patch["g"]), int(patch["f"])
        n_o = len(facts.get(str(o), []))
        if o >= k or g >= n_o or f >= n_o:
            raise RuntimeInputBuildError(f"imp patch index out of bounds: {patch}")

    conf_map: Dict[Tuple[int, int, int], int] = {}
    for patch in conf_patch:
        o, g, f = int(patch["o"]), int(patch["g"]), int(patch["f"])
        n_o = len(facts.get(str(o), []))
        if o >= k or g >= n_o or f >= n_o:
            raise RuntimeInputBuildError(f"conf patch index out of bounds: {patch}")
        conf_map[(o, g, f)] = int(patch["value"])

    for (o, g, f), val in conf_map.items():
        if conf_map.get((o, f, g), val) != val:
            raise RuntimeInputBuildError(f"conf matrix is not symmetric at object={o}, g={g}, f={f}")


def build_truthfinder_runtime_input_from_state(
    *,
    input_text: str,
    sentence_id: str,
    session_id: str,
    keywords: Sequence[str],
    results: Mapping[str, Mapping[str, Any]],
    cfg: TruthFinderConfig,
    schema_path: str | Path = "truthfinder_runtime_input_schema.json",
    normalized_by_model: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
    model_ids: Optional[Sequence[str]] = None,
    truthfinder_path: str | Path | None = None,
    normalize_path: str | Path | None = None,
    app_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Build runtime_input dict from app state + normalization/TruthFinder intermediates.

    Notes:
    - facts: built from normalized candidates, stable in-order dedup, fallback '(空)' when object has no candidates at all.
    - top1_choice: stores fact index (not text); -1 means no candidate for that model/object.
    - q16: uses deterministic half-up rounding with scale=65536.
    - hashes: SHA256 over stable JSON serialization (sorted keys + compact separators).
    """
    schema_file = Path(schema_path)
    if not schema_file.exists():
        raise FileNotFoundError(f"schema template not found: {schema_file}")

    template = _load_json_with_fallbacks(schema_file)
    data: Dict[str, Any] = copy.deepcopy(template)

    kws = [str(k).strip() for k in keywords if str(k).strip()]
    models = list(model_ids) if model_ids is not None else list(results.keys())

    if not models:
        raise RuntimeInputBuildError("model_ids is empty")

    # 1) normalized candidates per model+keyword
    norm_cands: Dict[str, Dict[str, List[str]]] = {}
    for m in models:
        norm_cands[m] = _normalize_candidates_for_model(
            model=m,
            keywords=kws,
            results=results,
            topn_candidates=cfg.topn_candidates,
            normalized_by_model=normalized_by_model,
        )

    # 2) object-level facts + support + top1(text)
    facts_by_o: Dict[int, List[str]] = {}
    support: Dict[Tuple[int, int, int], float] = {}  # (o, f, w) -> weight
    top1_text: Dict[str, Dict[Tuple[str, str], str]] = {m: {} for m in models}

    for o, kw in enumerate(kws):
        # Stable order: model order then candidate order
        all_facts: List[str] = []
        seen = set()
        for m in models:
            for cand in norm_cands[m].get(kw, []):
                if cand not in seen:
                    seen.add(cand)
                    all_facts.append(cand)

        if not all_facts:
            all_facts = [FALLBACK_FACT]
            for m in models:
                top1_text[m][(sentence_id, kw)] = FALLBACK_FACT
            for w, _m in enumerate(models):
                support[(o, 0, w)] = 1.0
        else:
            fact_to_idx = {f: i for i, f in enumerate(all_facts)}
            for w, m in enumerate(models):
                cands = norm_cands[m].get(kw, [])
                if not cands:
                    continue
                top1_text[m][(sentence_id, kw)] = cands[0]
                ws = _candidate_weights(len(cands), cfg.cand_decay)
                for cand, wgt in zip(cands, ws):
                    f_idx = fact_to_idx[cand]
                    support[(o, f_idx, w)] = support.get((o, f_idx, w), 0.0) + float(wgt)

        facts_by_o[o] = all_facts

    # 3) top1 choice index mapping
    top1_choice_idx: Dict[str, Dict[str, int]] = {}
    for m in models:
        per_obj: Dict[str, int] = {}
        for o, kw in enumerate(kws):
            facts = facts_by_o[o]
            top1 = top1_text[m].get((sentence_id, kw))
            if top1 is None:
                per_obj[str(o)] = -1
            else:
                try:
                    per_obj[str(o)] = facts.index(top1)
                except ValueError as ex:
                    raise RuntimeInputBuildError(
                        f"top1 fact text not found in facts list: model={m}, o={o}, fact={top1}"
                    ) from ex
        top1_choice_idx[m] = per_obj

    # 4) dep_avg patch from TruthFinder rho(top1)
    rho = compute_rho_dependency_top1(models, top1_text)
    dep_avg_q16: List[Dict[str, Any]] = []
    for w, m in enumerate(models):
        vals = [rho.get((m, u), 0.0) for u in models if u != m]
        dep_avg = (sum(vals) / len(vals)) if vals else 0.0
        dep_avg_q16.append({"w": w, "value": _float_to_q16_str(dep_avg)})

    # 5) implication/conflict patches per object
    imp_patch: List[Dict[str, Any]] = []
    conf_patch: List[Dict[str, Any]] = []

    for o, facts in facts_by_o.items():
        imp = build_implication_matrix(facts, sim_threshold=cfg.imp_sim_threshold)
        conf = build_conflict_matrix(facts, sim_threshold=cfg.conflict_sim_threshold)

        f_to_idx = {f: i for i, f in enumerate(facts)}
        for (g, f), val in sorted(imp.items(), key=lambda x: (f_to_idx[x[0][0]], f_to_idx[x[0][1]])):
            if val > 0:
                imp_patch.append(
                    {"o": o, "g": f_to_idx[g], "f": f_to_idx[f], "value": _float_to_q16_str(float(val))}
                )

        for (g, f), val in sorted(conf.items(), key=lambda x: (f_to_idx[x[0][0]], f_to_idx[x[0][1]])):
            if val > 0:
                conf_patch.append(
                    {"o": o, "g": f_to_idx[g], "f": f_to_idx[f], "value": _float_to_q16_str(float(val))}
                )

    # 6) support patch (non-zero only), stable sort o,f,w
    support_patch = [
        {"o": o, "f": f, "w": w, "value": _float_to_q16_str(v)}
        for (o, f, w), v in sorted(support.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]))
        if v > 0
    ]

    # 7) fill runtime/objects/facts/top1/params
    runtime = data.setdefault("runtime", {})
    runtime["session_id"] = session_id
    runtime["sentence_id"] = sentence_id
    runtime["input_text"] = input_text
    runtime["K"] = len(kws)
    runtime["model_ids"] = list(models)

    data["objects"] = [{"o": o, "keyword": kw, "is_effective": True} for o, kw in enumerate(kws)]
    data["facts"] = {str(o): facts for o, facts in facts_by_o.items()}
    data["top1_choice"] = top1_choice_idx

    data["params_q16"] = {
        "t0": _float_to_q16_str(cfg.t0),
        "beta": _float_to_q16_str(cfg.beta),
        "gamma": _float_to_q16_str(cfg.gamma),
        "alpha_imp": _float_to_q16_str(cfg.alpha_imp),
        "alpha_conflict": _float_to_q16_str(cfg.alpha_conflict),
        "cand_decay": _float_to_q16_str(cfg.cand_decay),
        "min_tau_scale": _float_to_q16_str(cfg.min_tau_scale),
    }

    params_meta = data.setdefault("params_meta", {})
    params_meta["topn_candidates"] = cfg.topn_candidates
    params_meta["imp_sim_threshold"] = cfg.imp_sim_threshold
    params_meta["conflict_sim_threshold"] = cfg.conflict_sim_threshold
    params_meta["delta"] = cfg.delta
    params_meta["max_iter"] = cfg.max_iter
    params_meta["support_mode"] = runtime.get("support_mode", "soft_candidates")

    facts_meta = data.setdefault("facts_meta", {})
    facts_meta["max_candidates_per_object"] = int(data.get("shape", {}).get("N_MAX", 12))

    patches = data.setdefault("patches", {})
    patches["dep_avg_patch"] = dep_avg_q16
    patches["support_patch"] = support_patch
    patches["imp_weight_patch"] = imp_patch
    patches["conf_weight_patch"] = conf_patch

    # 8) provenance + hashes
    repo_root = schema_file.parent
    truthfinder_file = Path(truthfinder_path) if truthfinder_path is not None else (repo_root / "TruthFinder.py")
    normalize_file = Path(normalize_path) if normalize_path is not None else (repo_root / "normalize.py")
    app_file = Path(app_path) if app_path is not None else (repo_root / "app.py")

    for fp in (truthfinder_file, normalize_file, app_file):
        if not fp.exists():
            raise FileNotFoundError(f"required source file not found: {fp}")

    provenance = data.setdefault("provenance", {})
    provenance["truthfinder_version"] = f"TruthFinder.py@{_file_sha256(truthfinder_file)[:12]}"
    provenance["normalize_version"] = f"normalize.py@{_file_sha256(normalize_file)[:12]}"
    provenance["app_version"] = f"app.py@{_file_sha256(app_file)[:12]}"
    provenance["generator_version"] = GENERATOR_VERSION

    facts_payload = {"facts": data["facts"]}
    imp_payload = {"imp_weight_patch": imp_patch}
    conf_payload = {"conf_weight_patch": conf_patch}
    support_payload = {"support_patch": support_patch}
    top1_payload = {"top1_choice": data["top1_choice"]}
    normalize_payload = {"normalized_candidates": norm_cands}
    sanitized_results = _sanitize_results_for_hash(results)
    input_payload = {
        "session_id": session_id,
        "sentence_id": sentence_id,
        "input_text": input_text,
        "keywords": kws,
        "results": sanitized_results,
    }

    provenance["facts_hash"] = _sha256_hex(facts_payload)
    provenance["imp_hash"] = _sha256_hex(imp_payload)
    provenance["conf_hash"] = _sha256_hex(conf_payload)
    provenance["support_hash"] = _sha256_hex(support_payload)
    provenance["top1_hash"] = _sha256_hex(top1_payload)
    provenance["normalize_hash"] = _sha256_hex(normalize_payload)
    provenance["input_hash"] = _sha256_hex(input_payload)

    _coerce_patch_indices_to_int(data)
    _validate_runtime_input(data)
    return data


def save_runtime_input_json(path: str | Path, data: Mapping[str, Any]) -> None:
    out = Path(path)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")