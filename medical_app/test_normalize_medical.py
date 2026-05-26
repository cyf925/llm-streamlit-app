from __future__ import annotations

import pytest

from medical_app.normalize_medical import (
    build_medical_fact_table,
    normalize_medical_fact,
    normalize_model_medical_output,
)


def _override_rules(result: dict) -> set[str]:
    return {
        item["rule"]
        for item in result.get("patches", {}).get("safety_overrides", [])
    }


def test_current_symptoms_not_treated_as_conditional() -> None:
    assert "胸痛" in normalize_medical_fact("risk_signal", "出现胸痛")
    assert "高热" in normalize_medical_fact("risk_signal", "出现高热")
    assert normalize_medical_fact("risk_signal", "伴随气短和出汗") == ["气短", "出汗"]
    assert normalize_medical_fact("risk_signal", "胸痛伴随左肩不适") == ["胸痛", "左肩或左臂不适"]


def test_current_prefix_is_not_treated_as_conditional() -> None:
    assert normalize_medical_fact("risk_signal", "当前胸痛") == ["胸痛"]
    assert normalize_medical_fact("risk_signal", "当前有胸痛") == ["胸痛"]
    assert normalize_medical_fact("risk_signal", "当前出现胸痛") == ["胸痛"]
    assert normalize_medical_fact("risk_signal", "当前伴随气短") == ["气短"]
    assert normalize_medical_fact("risk_signal", "当前胸闷并伴随出汗") == ["胸闷", "出汗"]
    assert normalize_medical_fact("risk_signal", "当出现胸痛时") == []
    assert normalize_medical_fact("risk_signal", "当伴随气短时") == []


def test_negated_risk_signals() -> None:
    assert normalize_medical_fact("risk_signal", "没有胸痛、没有气短、没有出汗") == []


def test_partial_negation_keeps_positive_signal() -> None:
    facts = normalize_medical_fact("risk_signal", "胸闷，但没有胸痛，也没有出汗")
    assert "胸闷" in facts
    assert "胸痛" not in facts
    assert "出汗" not in facts


def test_no_fever_but_cough_is_not_combo_low_risk() -> None:
    facts = normalize_medical_fact("low_risk_factor", "没有发烧，但是咳嗽")
    assert "无发热" in facts
    assert "无咳嗽" not in facts


def test_no_fever_cough_combo_is_recognized() -> None:
    assert normalize_medical_fact("low_risk_factor", "没有发烧咳嗽") == ["无发热", "无咳嗽"]
    assert normalize_medical_fact("low_risk_factor", "没发烧也没咳嗽") == ["无发热", "无咳嗽"]


def test_non_negating_exclusion_keeps_cause() -> None:
    assert "心血管相关" in normalize_medical_fact("possible_cause", "不能排除心脏问题")
    assert "心血管相关" in normalize_medical_fact("possible_cause", "需要排除心脏问题")


def test_true_exclusion_blocks_cause() -> None:
    assert "感染相关" not in normalize_medical_fact("possible_cause", "基本排除感染")
    assert "感染相关" not in normalize_medical_fact("possible_cause", "感染可能性低")


def test_conditional_emergency_after_current_outpatient() -> None:
    facts = normalize_medical_fact("urgency_level", "如胸痛加重应立即急诊，目前普通门诊即可")
    assert facts == ["普通门诊"]


def test_conditional_emergency_after_current_observation() -> None:
    facts = normalize_medical_fact("urgency_level", "如果出现胸痛应立即急诊，目前可以先观察")
    assert facts == ["短期观察"]


def test_current_judgment_with_conditional_tail_prefers_present_urgency() -> None:
    assert normalize_medical_fact("urgency_level", "目前普通门诊即可，如胸痛加重应立即急诊") == ["普通门诊"]
    assert normalize_medical_fact("urgency_level", "目前可以先观察，如果出现气短则立即急诊") == ["短期观察"]


def test_current_negative_statement_is_not_overridden_by_conditional_emergency() -> None:
    assert normalize_medical_fact("urgency_level", "当前没有明显危险信号，但若持续加重需急诊") != ["立即急诊"]


def test_conditional_risk_signal_is_filtered_but_current_risk_signal_is_kept() -> None:
    assert normalize_medical_fact("risk_signal", "当前胸痛") == ["胸痛"]
    assert normalize_medical_fact("risk_signal", "如果出现胸痛应立即急诊") == []


def test_numeric_high_fever_detection() -> None:
    assert "高热" in normalize_medical_fact("risk_signal", "发烧到39.5度")
    assert "高热" in normalize_medical_fact("risk_signal", "体温40度")
    assert "高热" not in normalize_medical_fact("risk_signal", "我今年39岁")

    result = normalize_model_medical_output({"user_explanation": "先注意休息"}, user_text="发烧到39.5度")
    assert "高热" in result["normalized"]["risk_signal"]


def test_numeric_high_fever_with_consciousness_change() -> None:
    facts = normalize_medical_fact("risk_signal", "体温40度，意识模糊")
    assert "高热" in facts
    assert "意识改变" in facts


def test_respiratory_and_infectious_expressions() -> None:
    cause_facts = normalize_medical_fact("possible_cause", "咳嗽咳痰三天，喉咙痛")
    assert {"呼吸系统相关", "感染相关"} & set(cause_facts)

    low_risk_facts = normalize_medical_fact("low_risk_factor", "鼻塞流鼻涕，没有明显发热")
    assert "无发热" in low_risk_facts


def test_urinary_mapping() -> None:
    assert "泌尿系统相关" in normalize_medical_fact("possible_cause", "尿频尿急尿痛")
    assert "尿血或尿潴留" in normalize_medical_fact("risk_signal", "尿血")


def test_skin_allergy_mapping() -> None:
    risk_facts = normalize_medical_fact("risk_signal", "全身风团，嘴唇肿，喘不上气")
    cause_facts = normalize_medical_fact("possible_cause", "全身风团，嘴唇肿，喘不上气")

    assert {"严重过敏反应", "喉头紧缩或面唇肿胀"} & set(risk_facts)
    assert "皮肤或过敏相关" in cause_facts


def test_ophthalmology_mapping() -> None:
    risk_facts = normalize_medical_fact("risk_signal", "突然一只眼看不清")
    cause_facts = normalize_medical_fact("possible_cause", "突然一只眼看不清")

    assert "视力突然下降" in risk_facts
    assert "眼科相关" in cause_facts


def test_ent_mapping() -> None:
    assert "耳鼻喉相关" in normalize_medical_fact("possible_cause", "耳痛、听力下降、咽痛")


def test_obstetric_mapping() -> None:
    risk_facts = normalize_medical_fact("risk_signal", "怀孕后下腹痛并见红")
    cause_facts = normalize_medical_fact("possible_cause", "怀孕后下腹痛并见红")

    assert "孕期出血或腹痛" in risk_facts
    assert "妇产/生殖相关" in cause_facts


def test_pediatric_emergency_mapping() -> None:
    result = normalize_model_medical_output({"user_explanation": "宝宝高烧，精神差，叫不醒"}, user_text="宝宝高烧，精神差，叫不醒")
    normalized = result["normalized"]

    assert "高热" in normalized["risk_signal"]
    assert "儿童精神差或反应差" in normalized["risk_signal"]
    assert {"儿科", "急诊"} & set(normalized["consult_department"])


def test_endocrine_mapping() -> None:
    assert "内分泌代谢相关" in normalize_medical_fact("possible_cause", "低血糖，出冷汗，心慌")


def test_toxic_mapping() -> None:
    risk_facts = normalize_medical_fact("risk_signal", "误服药物，吃了很多药")
    cause_facts = normalize_medical_fact("possible_cause", "误服药物，吃了很多药")

    assert "药物过量或中毒风险" in risk_facts
    assert "药物或中毒相关" in cause_facts


def test_respiratory_cause_can_infer_department_without_emergency() -> None:
    text = "咳嗽咳痰三天，喉咙痛，没有明显发热，没有气短。"
    result = normalize_model_medical_output({"user_explanation": "", "structured_analysis": {}}, user_text=text)
    normalized = result["normalized"]

    assert {"呼吸系统相关", "感染相关"} & set(normalized["possible_cause"])
    assert {"无发热", "无呼吸困难"} & set(normalized["low_risk_factor"])
    assert {"呼吸科", "全科/普通内科"} & set(normalized["consult_department"])
    assert normalized["urgency_level"] != ["立即急诊"]


def test_ent_cause_can_infer_department() -> None:
    text = "耳痛、听力下降、咽痛。"
    result = normalize_model_medical_output({"user_explanation": "", "structured_analysis": {}}, user_text=text)
    normalized = result["normalized"]

    assert "耳鼻喉相关" in normalized["possible_cause"]
    assert "耳鼻喉科" in normalized["consult_department"]


def test_urinary_cause_can_infer_department_without_emergency() -> None:
    text = "尿频尿急尿痛。"
    result = normalize_model_medical_output({"user_explanation": "", "structured_analysis": {}}, user_text=text)
    normalized = result["normalized"]

    assert "泌尿系统相关" in normalized["possible_cause"]
    assert "泌尿外科/肾内科" in normalized["consult_department"]
    assert normalized["urgency_level"] != ["立即急诊"]


def test_mild_skin_case_can_infer_department_without_emergency() -> None:
    text = "皮肤有点痒，起了少量红疹，没有呼吸困难。"
    result = normalize_model_medical_output({"user_explanation": "", "structured_analysis": {}}, user_text=text)
    normalized = result["normalized"]

    assert "皮肤或过敏相关" in normalized["possible_cause"]
    assert "皮肤科/变态反应科" in normalized["consult_department"]
    assert "无呼吸困难" in normalized["low_risk_factor"]
    assert normalized["urgency_level"] != ["立即急诊"]
    assert "severe-allergy-risk" not in _override_rules(result)


def test_ophthalmology_cause_can_infer_department() -> None:
    text = "眼睛红，视物模糊。"
    result = normalize_model_medical_output({"user_explanation": "", "structured_analysis": {}}, user_text=text)
    normalized = result["normalized"]

    assert "眼科相关" in normalized["possible_cause"]
    assert "眼科" in normalized["consult_department"]


def test_gynecology_cause_can_infer_department() -> None:
    text = "月经异常，下腹痛。"
    result = normalize_model_medical_output({"user_explanation": "", "structured_analysis": {}}, user_text=text)
    normalized = result["normalized"]

    assert "妇产/生殖相关" in normalized["possible_cause"]
    assert "妇产科" in normalized["consult_department"]


def test_endocrine_cause_can_infer_department() -> None:
    text = "低血糖，出冷汗，心慌。"
    result = normalize_model_medical_output({"user_explanation": "", "structured_analysis": {}}, user_text=text)
    normalized = result["normalized"]

    assert "内分泌代谢相关" in normalized["possible_cause"]
    assert "内分泌科" in normalized["consult_department"]


def test_severe_allergy_risk_override() -> None:
    text = "全身风团，嘴唇肿，喘不上气。"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)
    normalized = result["normalized"]

    assert {"严重过敏反应", "喉头紧缩或面唇肿胀"} & set(normalized["risk_signal"])
    assert normalized["danger_signal"] == ["存在明显危险信号"]
    assert normalized["urgency_level"] == ["立即急诊"]
    assert "皮肤或过敏相关" in normalized["possible_cause"]
    assert "急诊" in normalized["consult_department"]
    assert any(item["rule"] == "severe-allergy-risk" for item in result["patches"]["safety_overrides"])


def test_vision_loss_risk_override() -> None:
    text = "突然一只眼看不见，眼前有黑影。"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)
    normalized = result["normalized"]

    assert "视力突然下降" in normalized["risk_signal"]
    assert normalized["danger_signal"] == ["存在明显危险信号"]
    assert normalized["urgency_level"] == ["立即急诊"]
    assert "眼科相关" in normalized["possible_cause"]
    assert {"急诊", "眼科"} & set(normalized["consult_department"])
    assert any(item["rule"] == "vision-loss-risk" for item in result["patches"]["safety_overrides"])


def test_pregnancy_risk_override() -> None:
    text = "怀孕后下腹痛并见红。"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)
    normalized = result["normalized"]

    assert "孕期出血或腹痛" in normalized["risk_signal"]
    assert normalized["danger_signal"][0] in {"可能存在危险信号", "存在明显危险信号"}
    assert normalized["urgency_level"][0] in {"尽快线下就医", "立即急诊"}
    assert "妇产/生殖相关" in normalized["possible_cause"]
    assert "妇产科" in normalized["consult_department"]
    assert any(item["rule"] == "pregnancy-risk" for item in result["patches"]["safety_overrides"])


def test_pediatric_risk_override() -> None:
    text = "宝宝高烧，精神差，叫不醒。"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)
    normalized = result["normalized"]

    assert "高热" in normalized["risk_signal"]
    assert "儿童精神差或反应差" in normalized["risk_signal"]
    assert normalized["danger_signal"] == ["存在明显危险信号"]
    assert normalized["urgency_level"] == ["立即急诊"]
    assert {"儿科", "急诊"} & set(normalized["consult_department"])
    assert any(item["rule"] == "pediatric-risk" for item in result["patches"]["safety_overrides"])


def test_poisoning_risk_override() -> None:
    text = "误服药物，吃了很多药。"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)
    normalized = result["normalized"]

    assert "药物过量或中毒风险" in normalized["risk_signal"]
    assert "药物或中毒相关" in normalized["possible_cause"]
    assert normalized["danger_signal"] == ["存在明显危险信号"]
    assert normalized["urgency_level"] == ["立即急诊"]
    assert "急诊" in normalized["consult_department"]
    assert any(item["rule"] == "poisoning-risk" for item in result["patches"]["safety_overrides"])


def test_urinary_red_flag_override() -> None:
    text = "尿血，腰痛，发烧到39度。"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)
    normalized = result["normalized"]

    assert "尿血或尿潴留" in normalized["risk_signal"]
    assert "泌尿系统相关" in normalized["possible_cause"]
    assert normalized["urgency_level"][0] in {"尽快线下就医", "立即急诊"}
    assert "泌尿外科/肾内科" in normalized["consult_department"]
    assert any(item["rule"] == "urinary-red-flag" for item in result["patches"]["safety_overrides"])


def test_child_low_grade_fever_does_not_auto_trigger_pediatric_risk() -> None:
    text = "孩子有点低烧，精神还可以。"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)

    assert "儿科" in result["normalized"]["consult_department"]
    assert result["normalized"]["urgency_level"] != ["立即急诊"]
    assert "pediatric-risk" not in _override_rules(result)


def test_mild_rash_without_breathing_issue_does_not_trigger_severe_allergy_risk() -> None:
    text = "皮肤有点痒，起了少量红疹，没有呼吸困难。"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)
    normalized = result["normalized"]

    assert "皮肤或过敏相关" in normalized["possible_cause"]
    assert "皮肤科/变态反应科" in normalized["consult_department"]
    assert "无呼吸困难" in normalized["low_risk_factor"]
    assert normalized["urgency_level"] != ["立即急诊"]
    assert "severe-allergy-risk" not in _override_rules(result)


def test_chest_risk_with_rest_relief() -> None:
    text = "胸闷，休息后缓解"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)
    normalized = result["normalized"]

    assert "胸闷" in normalized["risk_signal"]
    assert "休息后缓解" in normalized["risk_signal"]
    assert normalized["danger_signal"][0] in {"存在明显危险信号", "可能存在危险信号"}
    assert "心血管相关" in normalized["possible_cause"]
    assert normalized["urgency_level"][0] not in {"短期观察", "信息不足"}


def test_strong_chest_risk_adds_cardiovascular_cause() -> None:
    text = "胸痛持续不缓解"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)
    normalized = result["normalized"]

    assert normalized["danger_signal"] == ["存在明显危险信号"]
    assert normalized["urgency_level"] == ["立即急诊"]
    assert "急诊" in normalized["consult_department"]
    assert "心血管相关" in normalized["possible_cause"]
    assert any(
        item["rule"] == "strong-chest-risk" and "possible_cause" in item["affected_objects"]
        for item in result["patches"]["safety_overrides"]
    )


def test_neuro_emergency_override() -> None:
    text = "突然说话不清，一侧肢体无力"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)
    normalized = result["normalized"]

    assert "突发无力或言语不清" in normalized["risk_signal"]
    assert "神经系统相关" in normalized["possible_cause"]
    assert normalized["urgency_level"] != ["短期观察"]
    assert {"急诊", "神经内科"} & set(normalized["consult_department"])


def test_high_fever_not_always_emergency() -> None:
    text = "高热"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)
    normalized = result["normalized"]

    assert "高热" in normalized["risk_signal"]
    assert "感染相关" in normalized["possible_cause"]
    assert normalized["urgency_level"] != ["立即急诊"]
    assert normalized["urgency_level"] == ["尽快线下就医"]


def test_high_fever_with_consciousness_change() -> None:
    text = "高热伴意识模糊"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)
    normalized = result["normalized"]

    assert "高热" in normalized["risk_signal"]
    assert "意识改变" in normalized["risk_signal"]
    assert "感染相关" in normalized["possible_cause"]
    assert normalized["urgency_level"] == ["立即急诊"]
    assert "急诊" in normalized["consult_department"]
    assert any(item["rule"] == "infection-with-consciousness-risk" for item in result["patches"]["safety_overrides"])


def test_abdominal_pain_override() -> None:
    text = "严重腹痛，持续不缓解，还想吐"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)
    normalized = result["normalized"]

    assert "剧烈腹痛" in normalized["risk_signal"]
    assert "恶心" in normalized["risk_signal"]
    assert "消化系统相关" in normalized["possible_cause"]
    assert normalized["urgency_level"][0] in {"尽快线下就医", "立即急诊"}


def test_severe_bleeding_override() -> None:
    text = "外伤后大量出血，止不住血"
    result = normalize_model_medical_output({"user_explanation": text}, user_text=text)
    normalized = result["normalized"]

    assert "严重出血" in normalized["risk_signal"]
    assert normalized["danger_signal"] == ["存在明显危险信号"]
    assert normalized["urgency_level"] == ["立即急诊"]
    assert "急诊" in normalized["consult_department"]


def test_negated_emergency_department() -> None:
    facts = normalize_medical_fact("consult_department", "不需要急诊，建议普通内科评估")
    assert "急诊" not in facts
    assert "全科/普通内科" in facts


def test_young_age_regex() -> None:
    assert normalize_medical_fact("low_risk_factor", "我20岁，最近压力大，经常熬夜") == [
        "年龄较轻",
        "近期压力大",
        "熬夜或睡眠不足",
    ]


def test_missing_structured_analysis_returns_fallbacks_and_warnings() -> None:
    result = normalize_model_medical_output(
        {"user_explanation": "需要更多信息，当前先无法判断。"},
        user_text="",
    )
    normalized = result["normalized"]

    assert normalized["danger_signal"] == ["信息不足"]
    assert normalized["urgency_level"] == ["信息不足"]
    assert normalized["possible_cause"] == ["原因不明"]
    assert normalized["consult_department"] == ["不确定"]
    assert normalized["risk_signal"] == []
    assert normalized["low_risk_factor"] == []
    assert any("structured_analysis missing" in warning for warning in result["warnings"])
    assert any("Missing field:" in warning for warning in result["warnings"])


def test_fallback_status_for_missing_possible_cause() -> None:
    result = normalize_model_medical_output({"structured_analysis": {}}, user_text="")
    assert result["normalized"]["possible_cause"] == ["原因不明"]
    assert result["field_status"]["possible_cause"]["missing"] is True
    assert result["field_status"]["possible_cause"]["used_fallback"] is True


def test_fallback_status_for_missing_consult_department() -> None:
    result = normalize_model_medical_output({"structured_analysis": {}}, user_text="")
    assert result["normalized"]["consult_department"] == ["不确定"]
    assert result["field_status"]["consult_department"]["used_fallback"] is True


def test_max_candidates_priority_truncation() -> None:
    model_output = {
        "structured_analysis": {
            "risk_signal": ["胸闷", "活动后加重", "严重出血", "意识改变", "心跳快或心慌"]
        }
    }
    result = normalize_model_medical_output(model_output, max_candidates_per_object=2)
    assert result["normalized"]["risk_signal"] == ["意识改变", "严重出血"]


def test_rows_include_source_fields() -> None:
    user_text = "胸闷，休息后缓解"
    result = normalize_model_medical_output(
        {"structured_analysis": {"urgency_level": "可以先观察几天"}},
        user_text=user_text,
    )
    row = next(item for item in result["rows"] if item["object_id"] == "urgency_level")
    assert "model_field_facts" in row
    assert "user_text_patch_facts" in row
    assert "final_normalized_facts" in row
    assert "normalized_facts" in row


def test_build_medical_fact_table_source_variants() -> None:
    normalized_all = {
        "模型A": normalize_model_medical_output(
            {
                "structured_analysis": {
                    "urgency_level": "可以先观察几天",
                    "possible_cause": ["压力大、熬夜相关"],
                }
            },
            user_text="胸闷，休息后缓解",
        )
    }
    normalized_table = build_medical_fact_table(normalized_all, source="normalized")
    model_table = build_medical_fact_table(normalized_all, source="from_model_fields")
    user_table = build_medical_fact_table(normalized_all, source="from_user_text")

    assert len(normalized_table) == 6
    assert len(model_table) == 6
    assert len(user_table) == 6

    urgency_model_row = next(row for row in model_table if row["object_id"] == "urgency_level")
    assert urgency_model_row["facts"] == ["短期观察"]

    risk_user_row = next(row for row in user_table if row["object_id"] == "risk_signal")
    assert "胸闷" in risk_user_row["facts"]

    with pytest.raises(ValueError):
        build_medical_fact_table(normalized_all, source="invalid")


def test_build_medical_fact_table_can_exclude_fallbacks_for_model_fields() -> None:
    normalized_all = {
        "模型A": normalize_model_medical_output(
            {"structured_analysis": {"urgency_level": "可以先观察几天"}},
            user_text="",
        )
    }
    keep_table = build_medical_fact_table(
        normalized_all,
        source="from_model_fields",
        exclude_fallbacks=False,
    )
    filtered_table = build_medical_fact_table(
        normalized_all,
        source="from_model_fields",
        exclude_fallbacks=True,
    )

    keep_row = next(row for row in keep_table if row["object_id"] == "possible_cause")
    filtered_row = next(row for row in filtered_table if row["object_id"] == "possible_cause")
    assert keep_row["facts"] == ["原因不明"]
    assert filtered_row["facts"] == []


def test_build_medical_fact_table_keeps_real_unknown_when_not_fallback() -> None:
    normalized_all = {
        "模型A": normalize_model_medical_output(
            {"structured_analysis": {"possible_cause": "原因不明"}},
            user_text="",
        )
    }
    result = normalized_all["模型A"]
    assert result["field_status"]["possible_cause"]["used_fallback"] is False

    filtered_table = build_medical_fact_table(
        normalized_all,
        source="from_model_fields",
        exclude_fallbacks=True,
    )
    row = next(row for row in filtered_table if row["object_id"] == "possible_cause")
    assert row["facts"] == ["原因不明"]


def test_model_and_user_text_sources_are_separate() -> None:
    user_text = "胸闷，休息后缓解，最近压力大，经常熬夜，没有发烧咳嗽"
    model_output = {
        "structured_analysis": {
            "urgency_level": "可以先观察几天",
            "possible_cause": ["压力大、熬夜相关"],
        }
    }
    result = normalize_model_medical_output(model_output, user_text=user_text)
    patches = result["patches"]

    assert patches["from_model_fields"]["urgency_level"] == ["短期观察"]
    assert "压力焦虑或睡眠相关" in patches["from_model_fields"]["possible_cause"]
    assert "胸闷" in patches["from_user_text"]["risk_signal"]
    assert "休息后缓解" in patches["from_user_text"]["risk_signal"]
    assert "无发热" in patches["from_user_text"]["low_risk_factor"]


def test_fact_table_shape_default_source() -> None:
    normalized_all = {
        "模型A": normalize_model_medical_output(
            {"structured_analysis": {"urgency_level": "普通门诊即可"}}
        )
    }
    table = build_medical_fact_table(normalized_all)
    assert len(table) == 6
    urgency_row = next(row for row in table if row["object_id"] == "urgency_level")
    assert urgency_row == {
        "model": "模型A",
        "object_id": "urgency_level",
        "object_label": "整体紧急程度",
        "facts": ["普通门诊"],
    }
