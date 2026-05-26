from __future__ import annotations

import json

import pytest

from medical_app.medical_truthfinder import (
    _build_candidate_map,
    _build_support_structures,
    EMPTY_FACT,
    MEDICAL_OBJECT_IDS,
    MedicalTruthFinderConfig,
    build_medical_model_facts,
    build_medical_relation_matrix,
    build_medical_zk_payload,
    explain_truth_per_medical_object,
    infer_model_family,
    make_medical_debug_jsonable,
    medical_fact_relation_score,
    medical_truthfinder_run,
    rank_models_by_trust,
    select_medical_top1_fact,
)
from medical_app.normalize_medical import normalize_model_medical_output


MODELS = ["模型A", "模型B", "模型C", "模型D"]


def _make_normalized_all() -> dict[str, dict]:
    return {
        "模型A": normalize_model_medical_output(
            {
                "structured_analysis": {
                    "danger_signal": "存在明显危险信号",
                    "urgency_level": "立即急诊",
                    "possible_cause": ["心血管相关"],
                    "risk_signal": ["胸闷", "气短", "出汗"],
                    "low_risk_factor": ["年龄较轻"],
                    "consult_department": ["急诊", "心内科"],
                }
            }
        ),
        "模型B": normalize_model_medical_output(
            {
                "structured_analysis": {
                    "danger_signal": "可能存在危险信号",
                    "urgency_level": "尽快线下就医",
                    "possible_cause": ["原因不明"],
                    "risk_signal": ["意识改变", "胸痛"],
                    "low_risk_factor": ["近期压力大", "熬夜或睡眠不足"],
                    "consult_department": ["神经内科"],
                }
            }
        ),
        "模型C": normalize_model_medical_output(
            {
                "structured_analysis": {
                    "danger_signal": "暂未发现明显危险信号",
                    "urgency_level": "短期观察",
                    "possible_cause": ["神经系统相关", "感染相关"],
                    "risk_signal": ["高热", "严重头痛"],
                    "low_risk_factor": ["无发热"],
                    "consult_department": ["全科/普通内科"],
                }
            }
        ),
        "模型D": normalize_model_medical_output({"structured_analysis": {}}),
    }


def _empty_cand_map(case_id: str) -> dict[tuple[str, str], list[str]]:
    return {(case_id, object_id): [EMPTY_FACT] for object_id in MEDICAL_OBJECT_IDS}


def test_build_medical_model_facts_defaults_to_model_fields_without_fallbacks() -> None:
    normalized_all = _make_normalized_all()
    model_facts = build_medical_model_facts(normalized_all, MODELS)

    assert model_facts["模型A"]["urgency_level"] == ["立即急诊"]
    assert model_facts["模型A"]["risk_signal"] == ["胸闷", "气短", "出汗"]
    assert model_facts["模型D"]["possible_cause"] == []


def test_cfg_is_not_mutated_in_place() -> None:
    normalized_all = _make_normalized_all()
    cfg = MedicalTruthFinderConfig(support_mode="multi")
    medical_truthfinder_run(
        MODELS,
        "case_cfg",
        normalized_all,
        cfg=cfg,
        support_mode="zk_top1",
    )
    assert cfg.support_mode == "multi"


def test_missing_field_fallback_does_not_enter_model_facts() -> None:
    normalized_all = {
        "模型A": normalize_model_medical_output(
            {"structured_analysis": {"urgency_level": "普通门诊"}}
        )
    }
    model_facts = build_medical_model_facts(normalized_all, ["模型A"])

    assert model_facts["模型A"]["possible_cause"] == []
    assert model_facts["模型A"]["consult_department"] == []


def test_multi_mode_support_keeps_all_multi_facts() -> None:
    normalized_all = _make_normalized_all()
    _, _, _, debug = medical_truthfinder_run(
        MODELS,
        "case_multi_support",
        normalized_all,
        support_mode="multi",
        return_debug=True,
    )

    obj = ("case_multi_support", "risk_signal")
    support = debug["support"]
    assert support[(obj, "胸闷")]["模型A"] == pytest.approx(1.0 / 3.0)
    assert support[(obj, "气短")]["模型A"] == pytest.approx(1.0 / 3.0)
    assert support[(obj, "出汗")]["模型A"] == pytest.approx(1.0 / 3.0)


def test_multi_mode_rho_uses_all_supported_facts_not_only_top1() -> None:
    normalized_all = {
        "模型A": normalize_model_medical_output(
            {"structured_analysis": {"risk_signal": ["胸闷", "气短"]}}
        ),
        "模型B": normalize_model_medical_output(
            {"structured_analysis": {"risk_signal": ["胸闷", "出汗"]}}
        ),
    }
    _, _, _, debug = medical_truthfinder_run(
        ["模型A", "模型B"],
        "case_rho_multi",
        normalized_all,
        support_mode="multi",
        return_debug=True,
    )

    assert debug["top1_choice"]["模型A"][("case_rho_multi", "risk_signal")] == "气短"
    assert debug["top1_choice"]["模型B"][("case_rho_multi", "risk_signal")] == "出汗"
    assert debug["rho"][("模型A", "模型B")] == pytest.approx(1.0 / 3.0)


def test_multi_mode_single_select_candidate_map_matches_support_semantics() -> None:
    case_id = "case_single_multi_consistency"
    obj = (case_id, "urgency_level")
    cfg = MedicalTruthFinderConfig(support_mode="multi")
    model_facts = {
        "模型A": {"urgency_level": ["普通门诊", "立即急诊"]},
        "模型B": {"urgency_level": ["普通门诊"]},
    }

    cand_map = _build_candidate_map(
        ["模型A", "模型B"],
        [obj],
        model_facts,
        cfg,
    )
    support, _top1_choice, _support_mask = _build_support_structures(
        ["模型A", "模型B"],
        [obj],
        model_facts,
        cfg,
    )

    assert "普通门诊" in cand_map[obj]
    assert "立即急诊" not in cand_map[obj]
    assert support[(obj, "普通门诊")] == {"模型A": 1.0, "模型B": 1.0}
    assert (obj, "立即急诊") not in support


def test_zk_top1_mode_keeps_one_support_and_one_top1_per_model_object() -> None:
    normalized_all = _make_normalized_all()
    _, _, _, debug = medical_truthfinder_run(
        MODELS,
        "case_001",
        normalized_all,
        support_mode="zk_top1",
        return_debug=True,
    )

    obj = ("case_001", "risk_signal")
    chosen = [
        fact
        for (support_obj, fact), by_model in debug["support"].items()
        if support_obj == obj and "模型A" in by_model
    ]
    assert chosen == ["气短"]
    assert debug["support_mask"]["模型A"][obj] == 1
    assert debug["top1_choice"]["模型A"][obj] == "气短"


def test_zk_top1_mode_candidate_map_collects_only_top1_facts() -> None:
    normalized_all = _make_normalized_all()
    _, _, cand_map, debug = medical_truthfinder_run(
        MODELS,
        "case_002",
        normalized_all,
        support_mode="zk_top1",
        return_debug=True,
    )

    obj = ("case_002", "risk_signal")
    assert cand_map[obj] == ["气短", "意识改变", "高热"]
    assert debug["top1_choice"]["模型A"][obj] == "气短"
    assert debug["top1_choice"]["模型B"][obj] == "意识改变"
    assert debug["top1_choice"]["模型C"][obj] == "高热"
    assert debug["top1_choice"]["模型D"][obj] is None


def test_zk_top1_mode_rho_still_uses_top1_choice() -> None:
    normalized_all = {
        "模型A": normalize_model_medical_output(
            {"structured_analysis": {"risk_signal": ["胸闷", "气短"]}}
        ),
        "模型B": normalize_model_medical_output(
            {"structured_analysis": {"risk_signal": ["气短", "胸闷"]}}
        ),
    }
    _, _, _, debug = medical_truthfinder_run(
        ["模型A", "模型B"],
        "case_rho_zk",
        normalized_all,
        support_mode="zk_top1",
        return_debug=True,
    )

    assert debug["top1_choice"]["模型A"][("case_rho_zk", "risk_signal")] == "气短"
    assert debug["top1_choice"]["模型B"][("case_rho_zk", "risk_signal")] == "气短"
    assert debug["rho"][("模型A", "模型B")] > 0.0


def test_single_model_run_has_change_history_and_abs_change_history() -> None:
    normalized_all = {"模型A": _make_normalized_all()["模型A"]}
    t_score, s_score, cand_map, debug = medical_truthfinder_run(
        ["模型A"],
        "case_single",
        normalized_all,
        return_debug=True,
    )

    assert "模型A" in t_score
    assert ("case_single", "urgency_level") in s_score
    assert ("case_single", "risk_signal") in cand_map
    assert isinstance(debug["change_history"], list)
    assert isinstance(debug["abs_change_history"], list)
    assert debug["iter_count"] >= 1


def test_string_fact_input_is_not_split_into_characters() -> None:
    assert select_medical_top1_fact("risk_signal", "胸痛") == "胸痛"


def test_risk_signal_top1_priority_prefers_severe_facts() -> None:
    assert select_medical_top1_fact("risk_signal", ["胸闷", "气短", "严重出血"]) == "严重出血"


def test_risk_signal_top1_priority_prefers_new_allergy_fact() -> None:
    assert select_medical_top1_fact("risk_signal", ["胸闷", "严重过敏反应", "气短"]) == "严重过敏反应"


def test_risk_signal_top1_priority_prefers_poisoning_over_chest_pain() -> None:
    assert select_medical_top1_fact("risk_signal", ["胸痛", "药物过量或中毒风险"]) == "药物过量或中毒风险"


def test_risk_signal_top1_priority_prefers_vision_loss_over_mild_fact() -> None:
    assert select_medical_top1_fact("risk_signal", ["视力突然下降", "胸闷"]) == "视力突然下降"


def test_possible_cause_top1_priority_prefers_poisoning_direction() -> None:
    assert select_medical_top1_fact("possible_cause", ["药物或中毒相关", "压力焦虑或睡眠相关"]) == "药物或中毒相关"


def test_consult_department_top1_priority_prefers_specialty_before_general() -> None:
    assert select_medical_top1_fact("consult_department", ["全科/普通内科", "妇产科"]) == "妇产科"
    assert select_medical_top1_fact("consult_department", ["全科/普通内科", "眼科"]) == "眼科"
    assert select_medical_top1_fact("consult_department", ["不确定", "急诊"]) == "急诊"


def test_top1_priority_ignores_invalid_fact_without_error() -> None:
    assert select_medical_top1_fact("risk_signal", ["不存在的fact", "严重过敏反应", "胸闷"]) == "严重过敏反应"


def test_infer_model_family_matches_expected_families() -> None:
    assert infer_model_family("qwen2.5:7b-instruct-q4_K_M") == "qwen"
    assert infer_model_family("gemma2:9b-instruct-q4_K_M") == "gemma"
    assert infer_model_family("koesn/mistral-7b-instruct:Q4_0") == "mistral"
    assert infer_model_family("phi3:mini") == "phi"
    assert infer_model_family("phi-4") == "phi"
    assert infer_model_family("gpt-4") == "gpt"
    assert infer_model_family("unknown-model") == "unknown"


def test_urgency_level_relation_is_negative_with_expected_strength() -> None:
    cfg = MedicalTruthFinderConfig()
    score = medical_fact_relation_score("urgency_level", "立即急诊", "短期观察", cfg)
    assert score == pytest.approx(-0.90)


def test_info_insufficient_has_weak_conflict_for_single_objects() -> None:
    cfg = MedicalTruthFinderConfig()
    assert medical_fact_relation_score("danger_signal", "信息不足", "存在明显危险信号", cfg) < 0.0
    assert medical_fact_relation_score("urgency_level", "信息不足", "立即急诊", cfg) < 0.0


def test_risk_signal_relation_is_zero_for_symptom_cooccurrence() -> None:
    cfg = MedicalTruthFinderConfig()
    assert medical_fact_relation_score("risk_signal", "胸痛", "气短", cfg) == 0.0


def test_possible_cause_unknown_vs_specific_is_negative() -> None:
    cfg = MedicalTruthFinderConfig()
    assert medical_fact_relation_score("possible_cause", "原因不明", "心血管相关", cfg) == pytest.approx(-0.35)


def test_relation_diagonal_is_zero_and_not_in_matrix() -> None:
    cfg = MedicalTruthFinderConfig()
    assert medical_fact_relation_score("urgency_level", "立即急诊", "立即急诊", cfg) == 0.0
    rel = build_medical_relation_matrix("urgency_level", ["立即急诊", "短期观察"], cfg)
    assert ("立即急诊", "立即急诊") not in rel


def test_medical_truthfinder_run_returns_debug_with_jsonable() -> None:
    normalized_all = _make_normalized_all()
    t_score, s_score, cand_map, debug = medical_truthfinder_run(
        MODELS,
        "case_debug",
        normalized_all,
        return_debug=True,
    )

    assert set(t_score) == set(MODELS)
    assert ("case_debug", "urgency_level") in s_score
    assert ("case_debug", "possible_cause") in cand_map
    assert "support" in debug
    assert "top1_choice" in debug
    assert "support_mask" in debug
    assert "relation_mats" in debug
    assert "dep_avg" in debug
    assert "jsonable" in debug


def test_model_coverage_and_effective_trust_are_reported() -> None:
    normalized_all = {
        "模型A": _make_normalized_all()["模型A"],
        "模型B": normalize_model_medical_output(
            {"structured_analysis": {"urgency_level": "普通门诊"}}
        ),
    }
    t_score, _, _, debug = medical_truthfinder_run(
        ["模型A", "模型B"],
        "case_coverage",
        normalized_all,
        support_mode="multi",
        return_debug=True,
    )

    assert debug["model_coverage"]["模型A"] > debug["model_coverage"]["模型B"]
    assert debug["effective_trust"]["模型B"] <= t_score["模型B"]
    assert "model_coverage" in debug["jsonable"]
    assert "effective_trust" in debug["jsonable"]


def test_make_medical_debug_jsonable_is_json_serializable() -> None:
    normalized_all = _make_normalized_all()
    _, _, _, debug = medical_truthfinder_run(
        MODELS,
        "case_jsonable",
        normalized_all,
        return_debug=True,
    )

    dumped = json.dumps(make_medical_debug_jsonable(debug), ensure_ascii=False)
    assert isinstance(dumped, str)


def test_explain_truth_empty_object_keeps_empty_result_and_no_watch_facts() -> None:
    case_id = "case_empty"
    cand_map = _empty_cand_map(case_id)
    rows = explain_truth_per_medical_object(case_id, {}, cand_map, support={})
    low_risk = next(row for row in rows if row["object_id"] == "low_risk_factor")

    assert low_risk["has_valid_result"] is False
    assert low_risk["selected_facts"] == []
    assert low_risk["watch_facts"] == []


def test_single_object_conservative_explanation_for_danger_signal() -> None:
    case_id = "case_cons_danger"
    cand_map = _empty_cand_map(case_id)
    cand_map[(case_id, "danger_signal")] = ["暂未发现明显危险信号", "可能存在危险信号"]
    s_score = {
        (case_id, "danger_signal"): {
            "暂未发现明显危险信号": 0.60,
            "可能存在危险信号": 0.59,
        }
    }

    rows = explain_truth_per_medical_object(case_id, s_score, cand_map, single_margin=0.03)
    row = next(item for item in rows if item["object_id"] == "danger_signal")
    assert row["selected_facts"] == ["暂未发现明显危险信号"]
    assert row["alternative_facts"] == ["可能存在危险信号"]
    assert row["risk_conservative_facts"] == ["可能存在危险信号"]


def test_single_object_conservative_explanation_for_urgency_level() -> None:
    case_id = "case_cons_urgency"
    cand_map = _empty_cand_map(case_id)
    cand_map[(case_id, "urgency_level")] = ["普通门诊", "尽快线下就医"]
    s_score = {
        (case_id, "urgency_level"): {
            "普通门诊": 0.60,
            "尽快线下就医": 0.59,
        }
    }

    rows = explain_truth_per_medical_object(case_id, s_score, cand_map, single_margin=0.03)
    row = next(item for item in rows if item["object_id"] == "urgency_level")
    assert row["selected_facts"] == ["普通门诊"]
    assert row["risk_conservative_facts"] == ["尽快线下就医"]


def test_risk_signal_watch_facts_are_reported_without_changing_selected_facts() -> None:
    case_id = "case_watch"
    cand_map = _empty_cand_map(case_id)
    cand_map[(case_id, "risk_signal")] = ["胸闷", "严重出血"]
    s_score = {
        (case_id, "risk_signal"): {
            "胸闷": 0.70,
            "严重出血": 0.50,
        }
    }
    support = {
        ((case_id, "risk_signal"), "胸闷"): {"模型A": 1.0},
        ((case_id, "risk_signal"), "严重出血"): {"模型B": 1.0},
    }

    rows = explain_truth_per_medical_object(
        case_id,
        s_score,
        cand_map,
        support=support,
        multi_threshold=0.55,
    )
    row = next(item for item in rows if item["object_id"] == "risk_signal")
    assert row["selected_facts"] == ["胸闷"]
    assert "严重出血" not in row["selected_facts"]
    assert row["watch_facts"] == ["严重出血"]


def test_new_high_risk_watch_facts_are_reported_without_changing_selected_facts() -> None:
    case_id = "case_watch_new"
    cand_map = _empty_cand_map(case_id)
    cand_map[(case_id, "risk_signal")] = ["胸闷", "严重过敏反应", "视力突然下降"]
    s_score = {
        (case_id, "risk_signal"): {
            "胸闷": 0.70,
            "严重过敏反应": 0.50,
            "视力突然下降": 0.49,
        }
    }
    support = {
        ((case_id, "risk_signal"), "胸闷"): {"模型A": 1.0},
        ((case_id, "risk_signal"), "严重过敏反应"): {"模型B": 1.0},
        ((case_id, "risk_signal"), "视力突然下降"): {"模型C": 1.0},
    }

    rows = explain_truth_per_medical_object(
        case_id=case_id,
        s_score=s_score,
        cand_map=cand_map,
        support=support,
        multi_threshold=0.55,
    )
    row = next(item for item in rows if item["object_id"] == "risk_signal")

    assert row["selected_facts"] == ["胸闷"]
    assert "严重过敏反应" not in row["selected_facts"]
    assert "视力突然下降" not in row["selected_facts"]
    assert "严重过敏反应" in row["watch_facts"]
    assert "视力突然下降" in row["watch_facts"]
    assert row["watch_reason"]


def test_rank_models_by_trust_sorts_descending() -> None:
    rank = rank_models_by_trust({"模型A": 0.4, "模型B": 0.8, "模型C": 0.6})
    assert rank == [("模型B", 0.8), ("模型C", 0.6), ("模型A", 0.4)]


def test_rank_models_by_trust_can_use_effective_trust() -> None:
    rank = rank_models_by_trust(
        {"A": 0.9, "B": 0.8},
        effective_trust={"A": 0.3, "B": 0.7},
        use_effective=True,
    )
    assert rank == [("B", 0.7), ("A", 0.3)]


def test_build_medical_zk_payload_shape_lengths_and_metadata() -> None:
    normalized_all = _make_normalized_all()
    _, _, cand_map, debug = medical_truthfinder_run(
        MODELS,
        "case_payload",
        normalized_all,
        support_mode="zk_top1",
        return_debug=True,
    )
    payload = build_medical_zk_payload(
        MODELS,
        "case_payload",
        cand_map,
        debug["top1_choice"],
        debug["support_mask"],
        debug["relation_mats"],
        debug["dep_avg"],
        n_max=8,
    )

    assert payload["shape"]["K"] == 6
    assert payload["shape"]["M"] == 4
    assert payload["shape"]["K_MAX"] == 10
    assert payload["shape"]["N_MAX"] == 8
    assert len(payload["top1_choice_flat"]) == 10 * 4
    assert len(payload["support_mask_flat"]) == 10 * 4
    assert len(payload["imp_flat"]) == 10 * 8 * 8
    assert len(payload["conf_flat"]) == 10 * 8 * 8
    assert len(payload["fact_count_by_object"]) == 10
    assert len(payload["is_effective_by_object"]) == 10
    assert len(payload["facts_padded"]) == 10
    assert len(payload["dep_avg_flat"]) == 4
    assert payload["public_output_layout_hint"]["public_len"] == 12


def test_build_medical_zk_payload_uses_object_major_flatten_order() -> None:
    normalized_all = _make_normalized_all()
    _, _, cand_map, debug = medical_truthfinder_run(
        MODELS,
        "case_flatten",
        normalized_all,
        support_mode="zk_top1",
        return_debug=True,
    )
    payload = build_medical_zk_payload(
        MODELS,
        "case_flatten",
        cand_map,
        debug["top1_choice"],
        debug["support_mask"],
        debug["relation_mats"],
        debug["dep_avg"],
        n_max=8,
    )

    urgency_index = list(MEDICAL_OBJECT_IDS).index("urgency_level")
    base = urgency_index * len(MODELS)
    assert payload["top1_choice_flat"][base + 0] == 0
    assert payload["top1_choice_flat"][base + 1] == 1
    assert payload["top1_choice_flat"][base + 2] == 2
    assert payload["top1_choice_flat"][base + 3] == -1
    assert payload["support_mask_flat"][base + 0] == 1
    assert payload["support_mask_flat"][base + 1] == 1
    assert payload["support_mask_flat"][base + 2] == 1
    assert payload["support_mask_flat"][base + 3] == 0


def test_build_medical_zk_payload_support_mask_must_be_zero_or_one() -> None:
    normalized_all = _make_normalized_all()
    _, _, cand_map, debug = medical_truthfinder_run(
        MODELS,
        "case_mask",
        normalized_all,
        support_mode="zk_top1",
        return_debug=True,
    )
    bad_support_mask = {
        model: dict(obj_map) for model, obj_map in debug["support_mask"].items()
    }
    bad_support_mask["模型A"] = dict(bad_support_mask["模型A"])
    bad_support_mask["模型A"][("case_mask", "urgency_level")] = 2

    with pytest.raises(ValueError, match="support_mask must be 0 or 1"):
        build_medical_zk_payload(
            MODELS,
            "case_mask",
            cand_map,
            debug["top1_choice"],
            bad_support_mask,
            debug["relation_mats"],
            debug["dep_avg"],
        )


def test_build_medical_zk_payload_dep_avg_must_be_in_range() -> None:
    normalized_all = _make_normalized_all()
    _, _, cand_map, debug = medical_truthfinder_run(
        MODELS,
        "case_dep",
        normalized_all,
        support_mode="zk_top1",
        return_debug=True,
    )
    bad_dep_avg = dict(debug["dep_avg"])
    bad_dep_avg["模型A"] = 1.5

    with pytest.raises(ValueError, match="dep_avg for model=模型A must be within"):
        build_medical_zk_payload(
            MODELS,
            "case_dep",
            cand_map,
            debug["top1_choice"],
            debug["support_mask"],
            debug["relation_mats"],
            bad_dep_avg,
        )


def test_build_medical_zk_payload_relation_score_must_be_in_range() -> None:
    normalized_all = _make_normalized_all()
    _, _, cand_map, debug = medical_truthfinder_run(
        MODELS,
        "case_rel",
        normalized_all,
        support_mode="zk_top1",
        return_debug=True,
    )
    bad_relation_mats = {
        obj: dict(rel_map) for obj, rel_map in debug["relation_mats"].items()
    }
    urgency_obj = ("case_rel", "urgency_level")
    facts = cand_map[urgency_obj]
    bad_relation_mats[urgency_obj] = dict(bad_relation_mats[urgency_obj])
    bad_relation_mats[urgency_obj][(facts[0], facts[1])] = 1.5

    with pytest.raises(ValueError, match="relation score must be within"):
        build_medical_zk_payload(
            MODELS,
            "case_rel",
            cand_map,
            debug["top1_choice"],
            debug["support_mask"],
            bad_relation_mats,
            debug["dep_avg"],
        )


def test_build_medical_zk_payload_warns_when_support_mode_is_not_zk_top1() -> None:
    normalized_all = _make_normalized_all()
    _, _, cand_map, debug = medical_truthfinder_run(
        MODELS,
        "case_warn",
        normalized_all,
        support_mode="zk_top1",
        return_debug=True,
    )
    payload = build_medical_zk_payload(
        MODELS,
        "case_warn",
        cand_map,
        debug["top1_choice"],
        debug["support_mask"],
        debug["relation_mats"],
        debug["dep_avg"],
        support_mode="multi",
        strict=False,
    )

    assert any("不是 ZK 对齐 top1 语义" in warning for warning in payload["warnings"])
    assert payload["zk_ready"] is False


def test_build_medical_zk_payload_is_zk_ready_when_top1_and_no_warnings() -> None:
    normalized_all = _make_normalized_all()
    _, _, cand_map, debug = medical_truthfinder_run(
        MODELS,
        "case_ready",
        normalized_all,
        support_mode="zk_top1",
        return_debug=True,
    )
    payload = build_medical_zk_payload(
        MODELS,
        "case_ready",
        cand_map,
        debug["top1_choice"],
        debug["support_mask"],
        debug["relation_mats"],
        debug["dep_avg"],
        support_mode="zk_top1",
    )

    assert payload["warnings"] == []
    assert payload["zk_ready"] is True
    assert payload["public_output_layout_hint"]["public_len"] == 2 + payload["shape"]["K_MAX"]


def test_no_valid_fact_model_has_zero_support_mask_negative_one_top1_and_no_empty_support() -> None:
    normalized_all = {
        "模型A": normalize_model_medical_output({"structured_analysis": {}}),
        "模型B": normalize_model_medical_output({"structured_analysis": {}}),
        "模型C": normalize_model_medical_output({"structured_analysis": {}}),
        "模型D": normalize_model_medical_output({"structured_analysis": {}}),
    }
    _, _, cand_map, debug = medical_truthfinder_run(
        MODELS,
        "case_none",
        normalized_all,
        support_mode="zk_top1",
        return_debug=True,
    )
    payload = build_medical_zk_payload(
        MODELS,
        "case_none",
        cand_map,
        debug["top1_choice"],
        debug["support_mask"],
        debug["relation_mats"],
        debug["dep_avg"],
        n_max=8,
    )

    urgency_index = list(MEDICAL_OBJECT_IDS).index("urgency_level")
    base = urgency_index * len(MODELS)
    assert cand_map[("case_none", "urgency_level")] == [EMPTY_FACT]
    assert ((("case_none", "urgency_level"), EMPTY_FACT) not in debug["support"])
    assert payload["support_mask_flat"][base + 0] == 0
    assert payload["top1_choice_flat"][base + 0] == -1
    assert payload["support_mask_flat"][base + 3] == 0
    assert payload["top1_choice_flat"][base + 3] == -1


def test_imp_and_conf_do_not_overlap() -> None:
    normalized_all = _make_normalized_all()
    _, _, cand_map, debug = medical_truthfinder_run(
        MODELS,
        "case_overlap",
        normalized_all,
        support_mode="zk_top1",
        return_debug=True,
    )
    payload = build_medical_zk_payload(
        MODELS,
        "case_overlap",
        cand_map,
        debug["top1_choice"],
        debug["support_mask"],
        debug["relation_mats"],
        debug["dep_avg"],
        n_max=8,
    )

    for imp, conf in zip(payload["imp_flat"], payload["conf_flat"]):
        assert not (imp > 0 and conf > 0)


def test_build_medical_zk_payload_raises_when_fact_count_exceeds_n_max() -> None:
    case_id = "case_overflow"
    cand_map = {
        (case_id, object_id): ([f"fact_{i}" for i in range(9)] if object_id == "risk_signal" else [EMPTY_FACT])
        for object_id in MEDICAL_OBJECT_IDS
    }
    top1_choice = {
        model: {(case_id, object_id): None for object_id in MEDICAL_OBJECT_IDS}
        for model in MODELS
    }
    support_mask = {
        model: {(case_id, object_id): 0 for object_id in MEDICAL_OBJECT_IDS}
        for model in MODELS
    }
    relation_mats = {(case_id, object_id): {} for object_id in MEDICAL_OBJECT_IDS}
    dep_avg = {model: 0.0 for model in MODELS}

    with pytest.raises(ValueError, match="risk_signal fact_count=9 exceeds n_max=8"):
        build_medical_zk_payload(
            MODELS,
            case_id,
            cand_map,
            top1_choice,
            support_mask,
            relation_mats,
            dep_avg,
            n_max=8,
        )
