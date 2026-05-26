from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import re
from typing import Any

from medical_app.normalize_medical import MEDICAL_OBJECTS, build_medical_fact_table


EMPTY_FACT = "(空)"
MEDICAL_OBJECT_IDS = tuple(schema.object_id for schema in MEDICAL_OBJECTS)
MEDICAL_OBJECT_LABELS = {schema.object_id: schema.label for schema in MEDICAL_OBJECTS}
MEDICAL_OBJECT_MODES = {schema.object_id: schema.mode for schema in MEDICAL_OBJECTS}
MEDICAL_OBJECT_OPTIONS = {schema.object_id: tuple(schema.options) for schema in MEDICAL_OBJECTS}
MEDICAL_FACT_SETS = {schema.object_id: set(schema.options) for schema in MEDICAL_OBJECTS}
SINGLE_SELECT_OBJECTS = {
    schema.object_id for schema in MEDICAL_OBJECTS if schema.mode == "single"
}
MULTI_SELECT_OBJECTS = set(MEDICAL_OBJECT_IDS) - SINGLE_SELECT_OBJECTS

_RISK_SIGNAL_TOP1_PRIORITY = (
    "药物过量或中毒风险",
    "严重过敏反应",
    "喉头紧缩或面唇肿胀",
    "严重出血",
    "意识改变",
    "抽搐",
    "突发无力或言语不清",
    "视力突然下降",
    "孕期出血或腹痛",
    "儿童精神差或反应差",
    "胸痛",
    "气短",
    "高热",
    "剧烈腹痛",
    "尿血或尿潴留",
    "严重头痛",
    "持续呕吐或腹泻",
    "脱水表现",
    "严重皮疹或快速扩散",
    "出汗",
    "恶心",
    "头晕",
    "胸闷",
    "活动后加重",
    "休息后缓解",
    "左肩或左臂不适",
    "心跳快或心慌",
)
_POSSIBLE_CAUSE_TOP1_PRIORITY = (
    "药物或中毒相关",
    "心血管相关",
    "神经系统相关",
    "呼吸系统相关",
    "感染相关",
    "妇产/生殖相关",
    "泌尿系统相关",
    "眼科相关",
    "皮肤或过敏相关",
    "消化系统相关",
    "内分泌代谢相关",
    "血液或免疫相关",
    "耳鼻喉相关",
    "肌肉骨骼相关",
    "压力焦虑或睡眠相关",
    "原因不明",
)
_CONSULT_DEPARTMENT_TOP1_PRIORITY = (
    "急诊",
    "心内科",
    "神经内科",
    "呼吸科",
    "妇产科",
    "儿科",
    "眼科",
    "泌尿外科/肾内科",
    "消化科",
    "皮肤科/变态反应科",
    "耳鼻喉科",
    "内分泌科",
    "血液科/风湿免疫科",
    "全科/普通内科",
    "骨科/康复科",
    "心理/精神心理科",
    "不确定",
)
_TOP1_PRIORITY = {
    "risk_signal": _RISK_SIGNAL_TOP1_PRIORITY,
    "possible_cause": _POSSIBLE_CAUSE_TOP1_PRIORITY,
    "consult_department": _CONSULT_DEPARTMENT_TOP1_PRIORITY,
}
_DANGER_LEVEL_ORDER = {
    "存在明显危险信号": 3,
    "可能存在危险信号": 2,
    "暂未发现明显危险信号": 1,
    "信息不足": 0,
}
_URGENCY_LEVEL_ORDER = {
    "立即急诊": 4,
    "尽快线下就医": 3,
    "普通门诊": 2,
    "短期观察": 1,
    "信息不足": 0,
}
_HIGH_RISK_WATCH_FACTS = {
    "药物过量或中毒风险",
    "严重过敏反应",
    "喉头紧缩或面唇肿胀",
    "严重出血",
    "意识改变",
    "抽搐",
    "突发无力或言语不清",
    "视力突然下降",
    "孕期出血或腹痛",
    "儿童精神差或反应差",
    "胸痛",
    "气短",
    "高热",
    "剧烈腹痛",
    "尿血或尿潴留",
    "严重头痛",
    "持续呕吐或腹泻",
    "脱水表现",
    "严重皮疹或快速扩散",
}
DEFAULT_MODEL_FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "qwen": ("qwen",),
    "gemma": ("gemma",),
    "mistral": ("mistral", "mixtral"),
    "llama": ("llama",),
    "phi": ("phi", "phi3", "phi4"),
    "deepseek": ("deepseek",),
    "gpt": ("gpt", "openai"),
}


def _matches_family_token(token: str, family: str, aliases: tuple[str, ...]) -> bool:
    for alias in aliases:
        if family == "phi":
            if token == alias or token.startswith(alias):
                return True
            continue
        if family == "gpt":
            if token == alias or token.startswith(alias):
                return True
            continue
        if token == alias or token.startswith(alias):
            return True
    return False


@dataclass
class MedicalTruthFinderConfig:
    t0: float = 0.75
    beta: float = 0.35
    gamma: float = 0.30
    alpha_imp: float = 0.20
    alpha_conflict: float = 0.10
    max_iter: int = 25
    early_stop: bool = True
    delta: float = 1e-4
    abs_delta: float = 1e-4
    min_iter: int = 2
    init_last_s: float = 0.5
    min_tau_scale: float = 0.40

    use_family_dependency: bool = True
    family_dep_same: float = 0.50
    family_dep_unknown: float = 0.10
    family_dep_different: float = 0.0
    model_family: dict[str, str] = field(default_factory=dict)

    use_trust_prior: bool = True
    trust_prior_default: float = 0.75
    trust_prior_by_model: dict[str, float] = field(default_factory=dict)
    trust_prior_strength: float = 2.0

    support_mode: str = "multi"
    empty_fact: str = EMPTY_FACT
    debug_relations: bool = True


def _sigmoid(x: float) -> float:
    if x >= 60:
        return 1.0
    if x <= -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _tau(t: float) -> float:
    t = min(max(float(t), 1e-6), 1.0 - 1e-6)
    return -math.log(1.0 - t)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _filter_valid_facts(object_id: str, facts: Any) -> list[str]:
    allowed = MEDICAL_FACT_SETS.get(object_id, set())
    if facts is None:
        raw_facts = []
    elif isinstance(facts, str):
        raw_facts = [facts]
    elif isinstance(facts, (list, tuple, set)):
        raw_facts = list(facts)
    else:
        raw_facts = [str(facts)]
    valid = [str(fact) for fact in raw_facts if str(fact) in allowed]
    return _dedupe_keep_order(valid)


def _default_model_fact_map(models: list[str]) -> dict[str, dict[str, list[str]]]:
    return {
        model: {object_id: [] for object_id in MEDICAL_OBJECT_IDS}
        for model in models
    }


def build_medical_model_facts(
    normalized_all: Any,
    models: list[str],
    source: str = "from_model_fields",
    exclude_fallbacks: bool = True,
    *,
    strict: bool = True,
) -> dict[str, dict[str, list[str]]]:
    model_facts = _default_model_fact_map(list(models or []))
    try:
        table = build_medical_fact_table(
            normalized_all or {},
            source=source,
            exclude_fallbacks=exclude_fallbacks,
        )
    except Exception:
        if strict:
            raise
        return model_facts

    for row in table:
        model = row.get("model")
        object_id = row.get("object_id")
        if model not in model_facts or object_id not in MEDICAL_FACT_SETS:
            continue
        model_facts[model][object_id] = _filter_valid_facts(object_id, row.get("facts", []))

    return model_facts


def select_medical_top1_fact(object_id: str, facts: Any) -> str | None:
    valid_facts = _filter_valid_facts(object_id, facts)
    if not valid_facts:
        return None
    if object_id in SINGLE_SELECT_OBJECTS:
        return valid_facts[0]

    priority = _TOP1_PRIORITY.get(object_id)
    if priority is None:
        priority = MEDICAL_OBJECT_OPTIONS.get(object_id, ())
    order = {fact: idx for idx, fact in enumerate(priority)}
    fallback_order = {
        fact: idx for idx, fact in enumerate(MEDICAL_OBJECT_OPTIONS.get(object_id, ()))
    }
    return min(valid_facts, key=lambda fact: (order.get(fact, 10_000), fallback_order.get(fact, 10_000)))


def _symmetric_negative_pair(
    g: str,
    f: str,
    scores: dict[tuple[str, str], float],
) -> float:
    if (g, f) in scores:
        return -scores[(g, f)]
    if (f, g) in scores:
        return -scores[(f, g)]
    return 0.0


def medical_fact_relation_score(
    object_id: str,
    g: str,
    f: str,
    cfg: MedicalTruthFinderConfig,
) -> float:
    if g == f:
        return 0.0

    if object_id == "danger_signal":
        return _symmetric_negative_pair(
            g,
            f,
            {
                ("存在明显危险信号", "暂未发现明显危险信号"): 0.90,
                ("可能存在危险信号", "暂未发现明显危险信号"): 0.55,
                ("信息不足", "存在明显危险信号"): 0.25,
                ("信息不足", "可能存在危险信号"): 0.20,
                ("信息不足", "暂未发现明显危险信号"): 0.20,
            },
        )

    if object_id == "urgency_level":
        return _symmetric_negative_pair(
            g,
            f,
            {
                ("立即急诊", "短期观察"): 0.90,
                ("立即急诊", "普通门诊"): 0.75,
                ("立即急诊", "尽快线下就医"): 0.30,
                ("尽快线下就医", "短期观察"): 0.60,
                ("尽快线下就医", "普通门诊"): 0.35,
                ("普通门诊", "短期观察"): 0.20,
                ("信息不足", "立即急诊"): 0.25,
                ("信息不足", "尽快线下就医"): 0.20,
                ("信息不足", "普通门诊"): 0.20,
                ("信息不足", "短期观察"): 0.20,
            },
        )

    if object_id == "possible_cause":
        if "原因不明" in {g, f} and g != f:
            return -0.35
        return 0.0

    if object_id == "consult_department":
        if "不确定" in {g, f} and g != f:
            return -0.35
        return 0.0

    if object_id in {"risk_signal", "low_risk_factor"}:
        return 0.0

    return 0.0


def build_medical_relation_matrix(
    object_id: str,
    facts: list[str],
    cfg: MedicalTruthFinderConfig,
) -> dict[tuple[str, str], float]:
    rel: dict[tuple[str, str], float] = {}
    unique_facts = _dedupe_keep_order([fact for fact in facts if fact])
    for g in unique_facts:
        for f in unique_facts:
            if g == f:
                continue
            score = medical_fact_relation_score(object_id, g, f, cfg)
            if score != 0.0:
                rel[(g, f)] = score
    return rel


def infer_model_family(model_name: str) -> str:
    normalized = (model_name or "").strip().lower()
    if not normalized:
        return "unknown"

    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    for family, aliases in DEFAULT_MODEL_FAMILY_ALIASES.items():
        for token in tokens:
            if _matches_family_token(token, family, aliases):
                return family
    return "unknown"


def _family_for_model(model: str, cfg: MedicalTruthFinderConfig) -> str:
    family = cfg.model_family.get(model)
    if family:
        return str(family).lower()
    return infer_model_family(model)


def compute_observed_rho(
    models: list[str],
    top1_choice: dict[str, dict[tuple[str, str], str | None]],
) -> dict[tuple[str, str], float]:
    choice_sets: dict[str, set[tuple[tuple[str, str], str]]] = {}
    for model in models:
        pairs: set[tuple[tuple[str, str], str]] = set()
        for obj, fact in (top1_choice.get(model) or {}).items():
            if fact is not None:
                pairs.add((obj, fact))
        choice_sets[model] = pairs

    rho: dict[tuple[str, str], float] = {}
    for model_a in models:
        for model_b in models:
            if model_a == model_b:
                continue
            set_a = choice_sets.get(model_a, set())
            set_b = choice_sets.get(model_b, set())
            rho[(model_a, model_b)] = _jaccard_pairs(set_a, set_b)
    return rho


def compute_observed_rho_from_support(
    models: list[str],
    support: dict[tuple[tuple[str, str], str], dict[str, float]],
) -> dict[tuple[str, str], float]:
    choice_sets: dict[str, set[tuple[tuple[str, str], str]]] = {model: set() for model in models}
    for (obj, fact), by_model in (support or {}).items():
        for model, weight in (by_model or {}).items():
            if model in choice_sets and float(weight) > 0.0:
                choice_sets[model].add((obj, fact))

    rho: dict[tuple[str, str], float] = {}
    for model_a in models:
        for model_b in models:
            if model_a == model_b:
                continue
            rho[(model_a, model_b)] = _jaccard_pairs(
                choice_sets.get(model_a, set()),
                choice_sets.get(model_b, set()),
            )
    return rho


def compute_model_coverage(
    models: list[str],
    objects: list[tuple[str, str]],
    support_mask: dict[str, dict[tuple[str, str], int]],
) -> dict[str, float]:
    denominator = len(objects)
    coverage: dict[str, float] = {}
    for model in models:
        if denominator == 0:
            coverage[model] = 0.0
            continue
        supported_object_count = sum(
            1
            for obj in objects
            if int((support_mask.get(model, {}) or {}).get(obj, 0)) == 1
        )
        coverage[model] = supported_object_count / denominator
    return coverage


def compute_dependency_with_family(
    models: list[str],
    observed_rho: dict[tuple[str, str], float],
    cfg: MedicalTruthFinderConfig,
) -> dict[tuple[str, str], float]:
    dependency: dict[tuple[str, str], float] = {}
    for model_a in models:
        for model_b in models:
            if model_a == model_b:
                continue
            observed = float(observed_rho.get((model_a, model_b), 0.0))
            if not cfg.use_family_dependency:
                dependency[(model_a, model_b)] = observed
                continue
            family_a = _family_for_model(model_a, cfg)
            family_b = _family_for_model(model_b, cfg)
            if family_a == family_b and family_a != "unknown":
                prior = float(cfg.family_dep_same)
            elif "unknown" in {family_a, family_b}:
                prior = float(cfg.family_dep_unknown)
            else:
                prior = float(cfg.family_dep_different)
            dependency[(model_a, model_b)] = max(observed, prior)
    return dependency


def _get_trust_prior(model: str, cfg: MedicalTruthFinderConfig) -> float:
    prior = cfg.trust_prior_by_model.get(model, cfg.trust_prior_default)
    return min(max(float(prior), 0.01), 0.99)


def _jaccard_pairs(
    set_a: set[tuple[tuple[str, str], str]],
    set_b: set[tuple[tuple[str, str], str]],
) -> float:
    union = set_a | set_b
    return (len(set_a & set_b) / len(union)) if union else 0.0


def _build_support_structures(
    models: list[str],
    objects: list[tuple[str, str]],
    model_facts: dict[str, dict[str, list[str]]],
    cfg: MedicalTruthFinderConfig,
) -> tuple[
    dict[tuple[tuple[str, str], str], dict[str, float]],
    dict[str, dict[tuple[str, str], str | None]],
    dict[str, dict[tuple[str, str], int]],
]:
    support: dict[tuple[tuple[str, str], str], dict[str, float]] = {}
    top1_choice: dict[str, dict[tuple[str, str], str | None]] = {
        model: {} for model in models
    }
    support_mask: dict[str, dict[tuple[str, str], int]] = {
        model: {} for model in models
    }

    for model in models:
        for obj in objects:
            object_id = obj[1]
            facts = _filter_valid_facts(object_id, model_facts.get(model, {}).get(object_id, []))
            if not facts:
                top1_choice[model][obj] = None
                support_mask[model][obj] = 0
                continue

            support_mask[model][obj] = 1
            chosen_top1 = select_medical_top1_fact(object_id, facts)
            top1_choice[model][obj] = chosen_top1

            if cfg.support_mode == "zk_top1":
                if chosen_top1 is None:
                    support_mask[model][obj] = 0
                    continue
                support.setdefault((obj, chosen_top1), {})
                support[(obj, chosen_top1)][model] = 1.0
                continue

            if object_id in SINGLE_SELECT_OBJECTS:
                chosen = facts[0]
                support.setdefault((obj, chosen), {})
                support[(obj, chosen)][model] = 1.0
                continue

            weight = 1.0 / len(facts)
            for fact in facts:
                support.setdefault((obj, fact), {})
                support[(obj, fact)][model] = support[(obj, fact)].get(model, 0.0) + weight

    return support, top1_choice, support_mask


def _build_candidate_map(
    models: list[str],
    objects: list[tuple[str, str]],
    model_facts: dict[str, dict[str, list[str]]],
    cfg: MedicalTruthFinderConfig,
) -> dict[tuple[str, str], list[str]]:
    cand_map: dict[tuple[str, str], list[str]] = {}
    for obj in objects:
        object_id = obj[1]
        candidates: list[str] = []
        for model in models:
            facts = _filter_valid_facts(object_id, model_facts.get(model, {}).get(object_id, []))
            if cfg.support_mode == "zk_top1":
                top1_fact = select_medical_top1_fact(object_id, facts)
                if top1_fact is not None:
                    candidates.append(top1_fact)
                continue
            if object_id in SINGLE_SELECT_OBJECTS:
                if facts:
                    candidates.append(facts[0])
                continue
            candidates.extend(facts)
        candidates = _dedupe_keep_order(candidates)
        if not candidates:
            candidates = [cfg.empty_fact]
        cand_map[obj] = candidates
    return cand_map


def make_medical_debug_jsonable(debug_info: dict[str, Any]) -> dict[str, Any]:
    def _obj_sort_key(item: tuple[tuple[str, str], Any]) -> tuple[str, str]:
        obj, _value = item
        case_id, object_id = obj
        return case_id, object_id

    def _support_sort_key(
        item: tuple[tuple[tuple[str, str], str], dict[str, float]]
    ) -> tuple[str, str, str]:
        (obj, fact), _by_model = item
        case_id, object_id = obj
        return case_id, object_id, fact

    def _model_obj_sort_key(item: tuple[tuple[str, str], Any]) -> tuple[str, str]:
        obj, _value = item
        case_id, object_id = obj
        return case_id, object_id

    objects = debug_info.get("objects", []) or []
    object_meta = {
        obj: {
            "case_id": obj[0],
            "object_id": obj[1],
            "object_label": MEDICAL_OBJECT_LABELS.get(obj[1], obj[1]),
        }
        for obj in objects
    }

    support_rows: list[dict[str, Any]] = []
    for (obj, fact), by_model in sorted(
        (debug_info.get("support", {}) or {}).items(),
        key=_support_sort_key,
    ):
        meta = object_meta.get(
            obj,
            {
                "case_id": obj[0],
                "object_id": obj[1],
                "object_label": MEDICAL_OBJECT_LABELS.get(obj[1], obj[1]),
            },
        )
        clean_support = {model: float(weight) for model, weight in (by_model or {}).items()}
        support_rows.append(
            {
                "case_id": meta["case_id"],
                "object_id": meta["object_id"],
                "object_label": meta["object_label"],
                "fact": fact,
                "support_by_model": clean_support,
                "support_weight": float(sum(clean_support.values())),
            }
        )

    top1_rows: list[dict[str, Any]] = []
    for model, obj_map in sorted((debug_info.get("top1_choice", {}) or {}).items()):
        for obj, fact in sorted((obj_map or {}).items(), key=_model_obj_sort_key):
            meta = object_meta.get(
                obj,
                {
                    "case_id": obj[0],
                    "object_id": obj[1],
                    "object_label": MEDICAL_OBJECT_LABELS.get(obj[1], obj[1]),
                },
            )
            top1_rows.append(
                {
                    "model": model,
                    "case_id": meta["case_id"],
                    "object_id": meta["object_id"],
                    "object_label": meta["object_label"],
                    "top1_fact": fact,
                }
            )

    support_mask_rows: list[dict[str, Any]] = []
    for model, obj_map in sorted((debug_info.get("support_mask", {}) or {}).items()):
        for obj, mask in sorted((obj_map or {}).items(), key=_model_obj_sort_key):
            meta = object_meta.get(
                obj,
                {
                    "case_id": obj[0],
                    "object_id": obj[1],
                    "object_label": MEDICAL_OBJECT_LABELS.get(obj[1], obj[1]),
                },
            )
            support_mask_rows.append(
                {
                    "model": model,
                    "case_id": meta["case_id"],
                    "object_id": meta["object_id"],
                    "object_label": meta["object_label"],
                    "support_mask": int(mask),
                }
            )

    relation_rows: list[dict[str, Any]] = []
    for obj, rel_map in sorted(
        (debug_info.get("relation_mats", {}) or {}).items(),
        key=_obj_sort_key,
    ):
        meta = object_meta.get(
            obj,
            {
                "case_id": obj[0],
                "object_id": obj[1],
                "object_label": MEDICAL_OBJECT_LABELS.get(obj[1], obj[1]),
            },
        )
        relations = [
            {"from": g, "to": f, "score": float(score)}
            for (g, f), score in sorted((rel_map or {}).items(), key=lambda item: (item[0][0], item[0][1]))
        ]
        relation_rows.append(
            {
                "case_id": meta["case_id"],
                "object_id": meta["object_id"],
                "object_label": meta["object_label"],
                "relations": relations,
            }
        )

    rho_rows = [
        {"model_a": model_a, "model_b": model_b, "value": float(value)}
        for (model_a, model_b), value in sorted(
            (debug_info.get("rho", {}) or {}).items(),
            key=lambda item: (item[0][0], item[0][1]),
        )
    ]
    dependency_rows = [
        {"model_a": model_a, "model_b": model_b, "value": float(value)}
        for (model_a, model_b), value in sorted(
            (debug_info.get("dependency", {}) or {}).items(),
            key=lambda item: (item[0][0], item[0][1]),
        )
    ]
    object_rows = [
        {
            "case_id": obj[0],
            "object_id": obj[1],
            "object_label": MEDICAL_OBJECT_LABELS.get(obj[1], obj[1]),
            "mode": MEDICAL_OBJECT_MODES.get(obj[1], ""),
        }
        for obj in objects
    ]

    return {
        "support": support_rows,
        "top1_choice": top1_rows,
        "support_mask": support_mask_rows,
        "relation_mats": relation_rows,
        "rho": rho_rows,
        "dependency": dependency_rows,
        "dep_avg": {model: float(value) for model, value in (debug_info.get("dep_avg", {}) or {}).items()},
        "model_coverage": {
            model: float(value) for model, value in (debug_info.get("model_coverage", {}) or {}).items()
        },
        "effective_trust": {
            model: float(value) for model, value in (debug_info.get("effective_trust", {}) or {}).items()
        },
        "t_history": [
            {model: float(value) for model, value in history.items()}
            for history in (debug_info.get("t_history", []) or [])
        ],
        "change_history": [float(value) for value in (debug_info.get("change_history", []) or [])],
        "abs_change_history": [float(value) for value in (debug_info.get("abs_change_history", []) or [])],
        "iter_count": int(debug_info.get("iter_count", 0)),
        "support_mode": debug_info.get("support_mode"),
        "source": debug_info.get("source"),
        "exclude_fallbacks": bool(debug_info.get("exclude_fallbacks", False)),
        "objects": object_rows,
        "model_facts": {
            model: {object_id: list(facts) for object_id, facts in obj_map.items()}
            for model, obj_map in (debug_info.get("model_facts", {}) or {}).items()
        },
    }


def medical_truthfinder_run(
    models: list[str],
    case_id: str,
    normalized_all: Any,
    cfg: MedicalTruthFinderConfig | None = None,
    *,
    source: str = "from_model_fields",
    exclude_fallbacks: bool = True,
    support_mode: str = "multi",
    strict: bool = True,
    return_debug: bool = False,
) -> Any:
    run_cfg = replace(cfg or MedicalTruthFinderConfig(), support_mode=support_mode)
    if run_cfg.support_mode not in {"multi", "zk_top1"}:
        raise ValueError(f"Invalid support_mode={run_cfg.support_mode!r}")

    models = list(models or [])
    objects = [(case_id, object_id) for object_id in MEDICAL_OBJECT_IDS]
    model_facts = build_medical_model_facts(
        normalized_all,
        models,
        source=source,
        exclude_fallbacks=exclude_fallbacks,
        strict=strict,
    )

    support, top1_choice, support_mask = _build_support_structures(
        models=models,
        objects=objects,
        model_facts=model_facts,
        cfg=run_cfg,
    )
    model_coverage = compute_model_coverage(models, objects, support_mask)
    cand_map = _build_candidate_map(
        models=models,
        objects=objects,
        model_facts=model_facts,
        cfg=run_cfg,
    )

    if run_cfg.support_mode == "multi":
        observed_rho = compute_observed_rho_from_support(models, support)
    else:
        observed_rho = compute_observed_rho(models, top1_choice)
    dependency = compute_dependency_with_family(models, observed_rho, run_cfg)
    dep_avg = {
        model: (
            sum(dependency.get((model, other), 0.0) for other in models if other != model)
            / max(len(models) - 1, 1)
        )
        for model in models
    }

    relation_mats = {
        obj: build_medical_relation_matrix(obj[1], cand_map[obj], run_cfg)
        for obj in objects
    }

    t_score = {model: float(run_cfg.t0) for model in models}
    last_s = {
        obj: {fact: float(run_cfg.init_last_s) for fact in cand_map[obj]}
        for obj in objects
    }
    s_score = {obj: {} for obj in objects}

    entries = {model: [] for model in models}
    for (obj, fact), by_model in support.items():
        for model, weight in by_model.items():
            if weight > 0.0:
                entries[model].append((obj, fact, float(weight)))

    t_history: list[dict[str, float]] = [dict(t_score)]
    change_history: list[float] = []
    abs_change_history: list[float] = []
    iter_count = 0

    for _ in range(run_cfg.max_iter):
        iter_count += 1
        old_t = dict(t_score)

        tau = {}
        for model in models:
            scale = 1.0 - run_cfg.gamma * dep_avg.get(model, 0.0)
            scale = max(run_cfg.min_tau_scale, scale)
            tau[model] = _tau(old_t[model]) * scale

        for obj in objects:
            facts = cand_map[obj]
            rel_mat = relation_mats.get(obj, {})
            base_sigma = {}
            for fact in facts:
                base_sigma[fact] = sum(
                    tau[model] * weight
                    for model, weight in (support.get((obj, fact), {}) or {}).items()
                )

            for fact in facts:
                rel_effect = 0.0
                for g in facts:
                    if g == fact:
                        continue
                    relation = rel_mat.get((g, fact), 0.0)
                    g_score = float(last_s.get(obj, {}).get(g, run_cfg.init_last_s))
                    if relation > 0.0:
                        rel_effect += run_cfg.alpha_imp * relation * g_score
                    elif relation < 0.0:
                        rel_effect += run_cfg.alpha_conflict * relation * g_score
                score = base_sigma[fact] + rel_effect
                s_score[obj][fact] = _sigmoid(run_cfg.beta * score)

        for obj in objects:
            last_s[obj] = dict(s_score.get(obj, {}))

        new_t = {}
        for model in models:
            model_entries = entries.get(model, [])
            if not model_entries:
                new_t[model] = old_t[model]
                continue

            numerator = 0.0
            denominator = 0.0
            for obj, fact, weight in model_entries:
                numerator += weight * float(s_score.get(obj, {}).get(fact, 0.0))
                denominator += weight

            if run_cfg.use_trust_prior:
                prior = _get_trust_prior(model, run_cfg)
                mu = max(0.0, float(run_cfg.trust_prior_strength))
                new_value = (mu * prior + numerator) / (mu + denominator)
            else:
                new_value = numerator / denominator if denominator > 0.0 else old_t[model]
            new_t[model] = min(max(float(new_value), 1e-6), 1.0 - 1e-6)

        old_vec = [old_t[model] for model in models]
        new_vec = [new_t[model] for model in models]
        max_abs_change = (
            max(abs(new_t[model] - old_t[model]) for model in models) if models else 0.0
        )
        if len(models) <= 1:
            change = max_abs_change
        else:
            change = max(0.0, 1.0 - _cosine(old_vec, new_vec))

        change_history.append(change)
        abs_change_history.append(max_abs_change)
        t_score = new_t
        t_history.append(dict(t_score))

        if (
            run_cfg.early_stop
            and iter_count >= run_cfg.min_iter
            and change < run_cfg.delta
            and max_abs_change < run_cfg.abs_delta
        ):
            break

    effective_trust = {
        model: float(t_score.get(model, 0.0)) * float(model_coverage.get(model, 0.0))
        for model in models
    }

    if not return_debug:
        return t_score, s_score, cand_map

    debug_info = {
        "support": support,
        "top1_choice": top1_choice,
        "support_mask": support_mask,
        "relation_mats": relation_mats if run_cfg.debug_relations else {},
        "dep_avg": dep_avg,
        "model_coverage": model_coverage,
        "effective_trust": effective_trust,
        "rho": observed_rho,
        "dependency": dependency,
        "t_history": t_history,
        "change_history": change_history,
        "abs_change_history": abs_change_history,
        "iter_count": iter_count,
        "objects": objects,
        "model_facts": model_facts,
        "support_mode": run_cfg.support_mode,
        "source": source,
        "exclude_fallbacks": exclude_fallbacks,
    }
    debug_info["jsonable"] = make_medical_debug_jsonable(debug_info)
    return t_score, s_score, cand_map, debug_info


def explain_truth_per_medical_object(
    case_id: str,
    s_score: dict[tuple[str, str], dict[str, float]],
    cand_map: dict[tuple[str, str], list[str]],
    support: dict[tuple[tuple[str, str], str], dict[str, float]] | None = None,
    top_k: int = 2,
    margin: float = 0.03,
    multi_threshold: float = 0.55,
    single_margin: float = 0.03,
    include_watch_facts: bool = True,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for object_id in MEDICAL_OBJECT_IDS:
        obj = (case_id, object_id)
        facts_conf = list((s_score.get(obj, {}) or {}).items())
        if not facts_conf:
            facts_conf = [(fact, 0.0) for fact in cand_map.get(obj, [EMPTY_FACT])]
        facts_conf.sort(key=lambda item: item[1], reverse=True)

        is_only_empty = (
            len(facts_conf) == 1
            and facts_conf[0][0] == EMPTY_FACT
            and float(sum(((support or {}).get((obj, EMPTY_FACT), {}) or {}).values())) == 0.0
        )

        selected: list[tuple[str, float]]
        alternative_facts: list[str] = []
        alternative_conf: list[float] = []
        risk_conservative_facts: list[str] = []
        risk_conservative_reason = ""
        watch_facts: list[str] = []
        watch_conf: list[float] = []
        watch_reason = ""

        if is_only_empty:
            selected = []
        elif MEDICAL_OBJECT_MODES[object_id] == "single":
            selected = facts_conf[:1]
            if len(facts_conf) > 1:
                top1, top2 = facts_conf[0], facts_conf[1]
                if float(top1[1]) - float(top2[1]) <= single_margin:
                    alternative_facts = [top2[0]]
                    alternative_conf = [float(top2[1])]
                    if object_id == "danger_signal":
                        top1_level = _DANGER_LEVEL_ORDER.get(top1[0], -1)
                        top2_level = _DANGER_LEVEL_ORDER.get(top2[0], -1)
                        risk_conservative_facts = [top2[0] if top2_level > top1_level else top1[0]]
                        risk_conservative_reason = "top1 与 top2 置信度接近，医疗场景下保留更高风险候选作为安全提示。"
                    elif object_id == "urgency_level":
                        top1_level = _URGENCY_LEVEL_ORDER.get(top1[0], -1)
                        top2_level = _URGENCY_LEVEL_ORDER.get(top2[0], -1)
                        risk_conservative_facts = [top2[0] if top2_level > top1_level else top1[0]]
                        risk_conservative_reason = "top1 与 top2 置信度接近，医疗场景下保留更高紧急度候选作为安全提示。"
        else:
            selected = [item for item in facts_conf if float(item[1]) >= multi_threshold]
            if not selected:
                selected = facts_conf[:1]

        selected_facts = {fact for fact, _ in selected}
        candidates = []
        for rank, (fact, confidence) in enumerate(facts_conf, start=1):
            support_by_model = dict((support or {}).get((obj, fact), {}) or {})
            support_weight = float(sum(support_by_model.values()))
            if (
                include_watch_facts
                and object_id == "risk_signal"
                and fact in _HIGH_RISK_WATCH_FACTS
                and support_weight > 0.0
                and fact not in selected_facts
            ):
                watch_facts.append(fact)
                watch_conf.append(float(confidence))
            candidates.append(
                {
                    "rank": rank,
                    "fact": fact,
                    "confidence": float(confidence),
                    "is_selected": fact in selected_facts,
                    "is_empty": fact == EMPTY_FACT,
                    "support_weight": support_weight,
                    "support_by_model": {
                        model: float(weight) for model, weight in support_by_model.items()
                    },
                }
            )
        if watch_facts:
            watch_reason = "该高风险信号虽未达到聚合入选阈值，但至少有模型支持，建议在最终提示中作为需关注候选展示。"

        out.append(
            {
                "object_id": object_id,
                "object_label": MEDICAL_OBJECT_LABELS[object_id],
                "mode": MEDICAL_OBJECT_MODES[object_id],
                "is_empty": is_only_empty,
                "has_valid_result": not is_only_empty,
                "selected_facts": [fact for fact, _ in selected],
                "selected_conf": [float(conf) for _, conf in selected],
                "alternative_facts": alternative_facts,
                "alternative_conf": alternative_conf,
                "risk_conservative_facts": risk_conservative_facts,
                "risk_conservative_reason": risk_conservative_reason,
                "watch_facts": watch_facts,
                "watch_conf": watch_conf,
                "watch_reason": watch_reason,
                "candidates": candidates,
            }
        )
    return out


def rank_models_by_trust(
    t_score: dict[str, float],
    *,
    effective_trust: dict[str, float] | None = None,
    use_effective: bool = False,
) -> list[tuple[str, float]]:
    score_map = effective_trust if use_effective and effective_trust is not None else t_score
    return sorted((score_map or {}).items(), key=lambda x: x[1], reverse=True)


def build_medical_zk_payload(
    models: list[str],
    case_id: str,
    cand_map: dict[tuple[str, str], list[str]],
    top1_choice: dict[str, dict[tuple[str, str], str | None]],
    support_mask: dict[str, dict[tuple[str, str], int]],
    relation_mats: dict[tuple[str, str], dict[tuple[str, str], float]],
    dep_avg: dict[str, float],
    *,
    k_max: int = 10,
    n_max: int = 8,
    iter_n: int = 15,
    support_mode: str = "zk_top1",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Build a ZK-friendly payload for later circuit/input conversion.

    This output is not the final circom witness input. It preserves the
    top1/support/relation layout in a circuit-friendly flat structure.

    Flatten order:
    1. top1_choice_flat / support_mask_flat:
       index = object_index * M + model_index
    2. imp_flat / conf_flat:
       object-major -> from_fact_index -> to_fact_index

    strict=False only downgrades limited non-structural issues into warnings.
    Structural errors still raise ValueError, including:
    - fact_count > n_max
    - support_mask not in {0, 1}
    - top1_choice out of range
    - imp/conf overlap
    - flattened length mismatches
    """
    warnings: list[str] = []
    if support_mode != "zk_top1":
        warnings.append("当前 payload 不是 ZK 对齐 top1 语义，不建议直接用于电路证明。")

    objects = list(MEDICAL_OBJECT_IDS)
    k = len(objects)
    if k > k_max:
        raise ValueError(f"K={k} exceeds k_max={k_max}")

    object_rows: list[dict[str, Any]] = []
    fact_index_map: dict[tuple[str, str], dict[str, int]] = {}
    fact_count_by_object: list[int] = []
    is_effective_by_object: list[int] = []
    facts_padded: list[list[str]] = []
    for object_index, object_id in enumerate(objects):
        obj = (case_id, object_id)
        facts = list(cand_map.get(obj, []))
        if len(facts) > n_max:
            raise ValueError(f"{object_id} fact_count={len(facts)} exceeds n_max={n_max}")
        fact_index_map[obj] = {fact: idx for idx, fact in enumerate(facts)}
        padded = facts + [""] * (n_max - len(facts))
        fact_count_by_object.append(len(facts))
        is_effective_by_object.append(1)
        facts_padded.append(padded)
        object_rows.append(
            {
                "object_index": object_index,
                "object_id": object_id,
                "object_label": MEDICAL_OBJECT_LABELS[object_id],
                "mode": MEDICAL_OBJECT_MODES[object_id],
                "is_effective": 1,
                "facts": list(facts),
                "fact_count": len(facts),
                "facts_padded": padded,
            }
        )
    for object_index in range(k, k_max):
        fact_count_by_object.append(0)
        is_effective_by_object.append(0)
        padded = [""] * n_max
        facts_padded.append(padded)
        object_rows.append(
            {
                "object_index": object_index,
                "object_id": "",
                "object_label": "",
                "mode": "",
                "is_effective": 0,
                "facts": [],
                "fact_count": 0,
                "facts_padded": padded,
            }
        )

    dep_avg_flat = []
    for model in models:
        dep_value = float(dep_avg.get(model, 0.0))
        if not (0.0 <= dep_value <= 1.0):
            message = f"dep_avg for model={model} must be within [0.0, 1.0], got {dep_value}"
            if strict:
                raise ValueError(message)
            warnings.append(message)
        dep_avg_flat.append(dep_value)

    top1_choice_flat: list[int] = []
    support_mask_flat: list[int] = []
    for object_index in range(k_max):
        if object_index >= k:
            for _model in models:
                top1_choice_flat.append(-1)
                support_mask_flat.append(0)
            continue
        object_id = objects[object_index]
        obj = (case_id, object_id)
        fact_count = fact_count_by_object[object_index]
        for model in models:
            fact = (top1_choice.get(model, {}) or {}).get(obj)
            mask = int((support_mask.get(model, {}) or {}).get(obj, 0))
            if mask not in {0, 1}:
                raise ValueError(
                    f"support_mask must be 0 or 1 for object={object_id}, model={model}, got {mask}"
                )
            index = fact_index_map.get(obj, {}).get(fact, -1) if fact is not None else -1
            if mask == 0:
                index = -1
            if mask == 1 and not (0 <= index < fact_count):
                raise ValueError(
                    f"Invalid top1 index for object={object_id}, model={model}: index={index}, fact_count={fact_count}"
                )
            top1_choice_flat.append(int(index))
            support_mask_flat.append(mask)

    imp_flat: list[float] = []
    conf_flat: list[float] = []
    for object_index in range(k_max):
        if object_index >= k:
            imp_flat.extend([0.0] * (n_max * n_max))
            conf_flat.extend([0.0] * (n_max * n_max))
            continue
        object_id = objects[object_index]
        obj = (case_id, object_id)
        facts = cand_map.get(obj, [])
        rel_map = relation_mats.get(obj, {}) or {}
        for g_index in range(n_max):
            for f_index in range(n_max):
                imp_value = 0.0
                conf_value = 0.0
                if g_index < len(facts) and f_index < len(facts):
                    if g_index == f_index:
                        score = float(rel_map.get((facts[g_index], facts[f_index]), 0.0))
                        if score != 0.0:
                            raise ValueError(
                                f"Relation diagonal must be zero for object={object_id}, fact={facts[g_index]}"
                            )
                    else:
                        score = float(rel_map.get((facts[g_index], facts[f_index]), 0.0))
                        if not (-1.0 <= score <= 1.0):
                            message = (
                                f"relation score must be within [-1.0, 1.0] for "
                                f"object={object_id}, from={facts[g_index]}, to={facts[f_index]}, got {score}"
                            )
                            if strict:
                                raise ValueError(message)
                            warnings.append(message)
                        if score > 0.0:
                            imp_value = score
                        elif score < 0.0:
                            conf_value = abs(score)
                if imp_value > 0.0 and conf_value > 0.0:
                    raise ValueError(
                        f"imp/conf overlap at object={object_id}, from_index={g_index}, to_index={f_index}"
                    )
                imp_flat.append(imp_value)
                conf_flat.append(conf_value)

    expected_model_slots = k_max * len(models)
    expected_relation_slots = k_max * n_max * n_max
    if len(top1_choice_flat) != expected_model_slots:
        raise ValueError(
            f"top1_choice_flat length mismatch: got {len(top1_choice_flat)}, expected {expected_model_slots}"
        )
    if len(support_mask_flat) != expected_model_slots:
        raise ValueError(
            f"support_mask_flat length mismatch: got {len(support_mask_flat)}, expected {expected_model_slots}"
        )
    if len(imp_flat) != expected_relation_slots:
        raise ValueError(
            f"imp_flat length mismatch: got {len(imp_flat)}, expected {expected_relation_slots}"
        )
    if len(conf_flat) != expected_relation_slots:
        raise ValueError(
            f"conf_flat length mismatch: got {len(conf_flat)}, expected {expected_relation_slots}"
        )

    for object_index in range(k_max):
        fact_count = fact_count_by_object[object_index]
        is_effective = is_effective_by_object[object_index]
        if object_index >= k:
            if fact_count != 0 or is_effective != 0:
                raise ValueError(f"Padding object at index={object_index} must be ineffective with fact_count=0")
        for model_index, model in enumerate(models):
            flat_index = object_index * len(models) + model_index
            top1_index = top1_choice_flat[flat_index]
            mask = support_mask_flat[flat_index]
            if object_index >= k:
                if top1_index != -1 or mask != 0:
                    raise ValueError(f"Padding object at index={object_index} must have top1=-1 and support_mask=0")
                continue
            if mask == 0 and top1_index != -1:
                raise ValueError(
                    f"When support_mask=0, top1_choice must be -1 for object={objects[object_index]}, model={model}"
                )
            if mask == 1 and not (0 <= top1_index < fact_count):
                raise ValueError(
                    f"When support_mask=1, top1_choice must be within fact_count for object={objects[object_index]}, model={model}"
                )

    for flat_index, (imp_value, conf_value) in enumerate(zip(imp_flat, conf_flat)):
        if imp_value != 0.0 and conf_value != 0.0:
            raise ValueError(f"imp/conf overlap at flat index={flat_index}")

    return {
        "shape": {
            "M": len(models),
            "K": k,
            "K_MAX": k_max,
            "N_MAX": n_max,
            "ITER_N": iter_n,
        },
        "flatten_order": {
            "top1_choice_flat": "object_major_o_then_model",
            "support_mask_flat": "object_major_o_then_model",
            "imp_conf_flat": "object_major_o_then_from_fact_then_to_fact",
        },
        "support_mode": support_mode,
        "warnings": warnings,
        "zk_ready": support_mode == "zk_top1" and len(warnings) == 0,
        "case_id": case_id,
        "models": list(models),
        "objects": object_rows,
        "fact_count_by_object": fact_count_by_object,
        "is_effective_by_object": is_effective_by_object,
        "facts_padded": facts_padded,
        "top1_choice_flat": top1_choice_flat,
        "support_mask_flat": support_mask_flat,
        "dep_avg_flat": dep_avg_flat,
        "dep_avg": {model: float(dep_avg.get(model, 0.0)) for model in models},
        "imp_flat": imp_flat,
        "conf_flat": conf_flat,
        "public_output_layout_hint": {
            "best_model_idx": 0,
            "best_model_score_q16": 1,
            "winning_fact_idx_by_object_start": 2,
            "winning_fact_idx_by_object_len": k_max,
            "public_len": 2 + k_max,
        },
    }


__all__ = [
    "MEDICAL_OBJECT_IDS",
    "MedicalTruthFinderConfig",
    "build_medical_model_facts",
    "build_medical_relation_matrix",
    "build_medical_zk_payload",
    "compute_model_coverage",
    "compute_observed_rho_from_support",
    "compute_dependency_with_family",
    "compute_observed_rho",
    "explain_truth_per_medical_object",
    "make_medical_debug_jsonable",
    "medical_fact_relation_score",
    "medical_truthfinder_run",
    "rank_models_by_trust",
    "select_medical_top1_fact",
]
