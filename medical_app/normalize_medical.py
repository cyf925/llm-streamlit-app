from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class MedicalObjectSchema:
    object_id: str
    label: str
    mode: str
    options: tuple[str, ...]
    fallback: tuple[str, ...]


@dataclass(frozen=True)
class Clause:
    raw: str
    normalized: str


MEDICAL_OBJECTS: tuple[MedicalObjectSchema, ...] = (
    MedicalObjectSchema(
        object_id="danger_signal",
        label="是否存在危险信号",
        mode="single",
        options=(
            "存在明显危险信号",
            "可能存在危险信号",
            "暂未发现明显危险信号",
            "信息不足",
        ),
        fallback=("信息不足",),
    ),
    MedicalObjectSchema(
        object_id="urgency_level",
        label="整体紧急程度",
        mode="single",
        options=(
            "立即急诊",
            "尽快线下就医",
            "普通门诊",
            "短期观察",
            "信息不足",
        ),
        fallback=("信息不足",),
    ),
    MedicalObjectSchema(
        object_id="possible_cause",
        label="可能原因方向",
        mode="multi",
        options=(
            "心血管相关",
            "呼吸系统相关",
            "神经系统相关",
            "消化系统相关",
            "感染相关",
            "肌肉骨骼相关",
            "泌尿系统相关",
            "皮肤或过敏相关",
            "耳鼻喉相关",
            "眼科相关",
            "妇产/生殖相关",
            "内分泌代谢相关",
            "血液或免疫相关",
            "药物或中毒相关",
            "压力焦虑或睡眠相关",
            "原因不明",
        ),
        fallback=("原因不明",),
    ),
    MedicalObjectSchema(
        object_id="risk_signal",
        label="关键风险信号",
        mode="multi",
        options=(
            "胸闷",
            "胸痛",
            "活动后加重",
            "休息后缓解",
            "左肩或左臂不适",
            "心跳快或心慌",
            "气短",
            "出汗",
            "恶心",
            "头晕",
            "突发无力或言语不清",
            "严重头痛",
            "高热",
            "意识改变",
            "剧烈腹痛",
            "严重出血",
            "抽搐",
            "视力突然下降",
            "严重过敏反应",
            "喉头紧缩或面唇肿胀",
            "持续呕吐或腹泻",
            "脱水表现",
            "尿血或尿潴留",
            "孕期出血或腹痛",
            "儿童精神差或反应差",
            "严重皮疹或快速扩散",
            "药物过量或中毒风险",
        ),
        fallback=(),
    ),
    MedicalObjectSchema(
        object_id="low_risk_factor",
        label="支持低风险或其他解释的因素",
        mode="multi",
        options=(
            "年龄较轻",
            "近期压力大",
            "熬夜或睡眠不足",
            "无发热",
            "无咳嗽",
            "无呼吸困难",
            "无意识改变",
            "症状较轻",
            "症状短暂且已缓解",
            "无持续加重",
            "无明显外伤",
            "休息后改善",
        ),
        fallback=(),
    ),
    MedicalObjectSchema(
        object_id="consult_department",
        label="建议咨询科室",
        mode="multi",
        options=(
            "急诊",
            "全科/普通内科",
            "心内科",
            "神经内科",
            "呼吸科",
            "消化科",
            "骨科/康复科",
            "心理/精神心理科",
            "泌尿外科/肾内科",
            "皮肤科/变态反应科",
            "耳鼻喉科",
            "眼科",
            "妇产科",
            "儿科",
            "内分泌科",
            "血液科/风湿免疫科",
            "不确定",
        ),
        fallback=("不确定",),
    ),
)

OBJECT_INDEX: dict[str, MedicalObjectSchema] = {
    schema.object_id: schema for schema in MEDICAL_OBJECTS
}

OBJECT_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "danger_signal": {
        "存在明显危险信号": (
            "存在明显危险信号",
            "明显危险信号",
            "红旗信号明显",
            "高度警惕",
            "严重危险信号",
            "危险信号明显",
        ),
        "可能存在危险信号": (
            "可能存在危险信号",
            "需要警惕",
            "不能排除风险",
            "需重视",
            "存在需要关注的信号",
            "风险不能完全排除",
            "需进一步排查",
            "需要进一步排查",
        ),
        "暂未发现明显危险信号": (
            "暂未发现明显危险信号",
            "未发现明显危险信号",
            "暂无明显危险信号",
            "无明显红旗信号",
            "暂不明显",
            "目前看不出明显危险信号",
            "无明显危险信号",
        ),
        "信息不足": (
            "信息不足",
            "无法判断",
            "需要更多信息",
            "不好判断",
            "暂时无法评估",
        ),
    },
    "urgency_level": {
        "立即急诊": (
            "立即急诊",
            "马上急诊",
            "急诊处理",
            "拨打急救",
            "急救服务",
            "立刻就医",
            "马上去急诊",
            "立即去急诊",
            "应立即急诊",
        ),
        "尽快线下就医": (
            "尽快线下就医",
            "尽快就医",
            "尽快去医院",
            "建议检查",
            "尽快做基础检查",
            "需要线下评估",
            "及时就医",
            "尽快门诊",
            "建议尽快评估",
            "尽快到医院",
            "线下就诊",
            "建议线下评估",
            "尽早就医",
            "尽早就诊",
            "到医院评估",
            "建议医院评估",
            "预约门诊",
        ),
        "普通门诊": (
            "普通门诊",
            "门诊就诊",
            "社区医院",
            "医院内科",
            "门诊即可",
            "普通内科门诊",
            "去门诊评估",
        ),
        "短期观察": (
            "短期观察",
            "可以先观察",
            "观察几天",
            "暂时观察",
            "休息观察",
            "调整作息观察",
            "先观察",
            "先休息观察",
        ),
        "信息不足": (
            "信息不足",
            "无法判断",
            "需要更多信息",
            "不好评估",
        ),
    },
    "possible_cause": {
        "心血管相关": (
            "心血管相关",
            "心脏",
            "心血管",
            "心内科",
            "冠心病",
            "心肌缺血",
            "心律失常",
            "心绞痛",
            "需要排除心脏问题",
            "心脏问题",
        ),
        "呼吸系统相关": (
            "呼吸系统相关",
            "呼吸",
            "肺",
            "气道",
            "哮喘",
            "肺炎",
            "支气管",
            "呼吸道",
            "过度换气",
            "咳嗽",
            "咳痰",
            "痰多",
            "咽痛",
            "喉咙痛",
            "嗓子疼",
            "鼻塞",
            "流鼻涕",
            "流涕",
            "打喷嚏",
            "喉咙不舒服",
            "上呼吸道",
            "支气管炎",
        ),
        "神经系统相关": (
            "神经系统相关",
            "神经",
            "脑",
            "中风",
            "卒中",
            "神经系统",
        ),
        "消化系统相关": (
            "消化系统相关",
            "胃",
            "反酸",
            "消化",
            "胃食管反流",
            "胃反流",
            "反流性食管炎",
            "烧心",
            "腹痛",
            "胃肠",
        ),
        "感染相关": (
            "感染相关",
            "感染",
            "发热",
            "发烧",
            "炎症",
            "病毒",
            "细菌",
            "畏寒",
            "寒战",
            "咽痛",
            "喉咙痛",
            "扁桃体发炎",
            "流感",
            "感冒",
            "病毒感染",
            "细菌感染",
            "炎症反应",
        ),
        "肌肉骨骼相关": (
            "肌肉骨骼相关",
            "肌肉",
            "肩颈",
            "劳损",
            "拉伤",
            "骨骼",
            "颈椎",
            "肌肉紧张",
            "胸壁疼痛",
            "肋间神经痛",
        ),
        "泌尿系统相关": (
            "泌尿系统相关",
            "尿频",
            "尿急",
            "尿痛",
            "排尿痛",
            "小便痛",
            "小便刺痛",
            "尿不尽",
            "尿血",
            "血尿",
            "肾结石",
            "尿路感染",
            "膀胱炎",
            "肾盂肾炎",
            "腰痛伴尿痛",
        ),
        "皮肤或过敏相关": (
            "皮肤或过敏相关",
            "皮疹",
            "红疹",
            "风团",
            "荨麻疹",
            "瘙痒",
            "过敏",
            "皮肤红肿",
            "药疹",
            "湿疹",
            "皮肤肿",
            "嘴唇肿",
            "眼睑肿",
        ),
        "耳鼻喉相关": (
            "耳鼻喉相关",
            "咽痛",
            "喉咙痛",
            "嗓子疼",
            "鼻塞",
            "流鼻涕",
            "流涕",
            "耳痛",
            "耳鸣",
            "听力下降",
            "吞咽痛",
            "扁桃体",
            "鼻窦炎",
        ),
        "眼科相关": (
            "眼科相关",
            "眼痛",
            "视物模糊",
            "视力下降",
            "眼前发黑",
            "眼睛红",
            "眼压高",
            "飞蚊",
            "闪光感",
            "看不清",
        ),
        "妇产/生殖相关": (
            "妇产/生殖相关",
            "怀孕",
            "孕期",
            "月经异常",
            "阴道出血",
            "下腹痛",
            "痛经",
            "白带异常",
            "外阴瘙痒",
            "产后",
            "见红",
        ),
        "内分泌代谢相关": (
            "内分泌代谢相关",
            "血糖高",
            "血糖低",
            "低血糖",
            "糖尿病",
            "甲状腺",
            "甲亢",
            "甲减",
            "多饮多尿",
            "体重明显下降",
            "代谢异常",
        ),
        "血液或免疫相关": (
            "血液或免疫相关",
            "贫血",
            "免疫",
            "风湿",
            "关节肿痛",
            "出血倾向",
            "紫癜",
            "反复感染",
            "白细胞",
            "血小板",
        ),
        "药物或中毒相关": (
            "药物或中毒相关",
            "吃错药",
            "药物过量",
            "误服",
            "中毒",
            "酒精中毒",
            "农药",
            "一氧化碳",
            "煤气中毒",
            "药物反应",
        ),
        "压力焦虑或睡眠相关": (
            "压力焦虑或睡眠相关",
            "压力",
            "焦虑",
            "熬夜",
            "睡眠不足",
            "自主神经",
            "植物神经",
            "心理",
            "睡不好",
            "功能性不适",
        ),
        "原因不明": (
            "原因不明",
            "不明确",
            "无法判断",
            "不好判断",
            "脱水",
        ),
    },
    "risk_signal": {
        "胸闷": (
            "胸闷",
            "胸口闷",
            "胸口有点闷",
            "胸口发闷",
            "胸部压迫",
            "胸口压迫",
            "心前区不适",
            "胸部压迫感",
            "胸部不适",
            "胸口不适",
            "胸堵",
            "胸口堵",
            "胸口发紧",
            "胸部发紧",
        ),
        "胸痛": (
            "胸痛",
            "胸口痛",
            "胸部疼痛",
            "压榨痛",
            "严重胸痛",
            "剧烈胸痛",
        ),
        "活动后加重": (
            "活动后加重",
            "运动后加重",
            "爬楼梯加重",
            "走快加重",
            "劳累后加重",
            "爬楼梯明显",
            "爬楼梯或者走快时明显",
            "走快时明显",
            "一活动就更明显",
            "活动时明显",
            "运动时明显",
            "走路时明显",
            "上楼明显",
            "劳累时明显",
        ),
        "休息后缓解": (
            "休息后缓解",
            "休息缓解",
            "停下来缓解",
            "休息后改善",
        ),
        "左肩或左臂不适": (
            "左肩或左臂不适",
            "左肩",
            "左臂",
            "左上肢",
            "肩膀酸",
            "左肩酸",
            "放射到左肩",
            "左手臂不适",
        ),
        "心跳快或心慌": (
            "心跳快或心慌",
            "心跳快",
            "心慌",
            "心悸",
            "心跳明显",
            "心律不齐",
            "心跳异常",
        ),
        "气短": (
            "气短",
            "呼吸困难",
            "喘不上气",
            "憋气",
            "明显气短",
            "喘不过气",
            "呼吸费劲",
        ),
        "出汗": (
            "出汗",
            "冷汗",
            "大汗",
        ),
        "恶心": (
            "恶心",
            "想吐",
            "呕吐",
        ),
        "头晕": (
            "头晕",
            "眩晕",
            "晕厥",
            "快晕倒",
        ),
        "突发无力或言语不清": (
            "突发无力或言语不清",
            "言语不清",
            "说话不清",
            "突发无力",
            "一侧无力",
            "偏瘫",
            "说话含糊",
            "说不清话",
            "一边手脚没力气",
            "半边身体无力",
            "一侧肢体无力",
        ),
        "严重头痛": (
            "严重头痛",
            "剧烈头痛",
            "爆炸样头痛",
            "头疼得厉害",
            "爆炸样疼痛",
        ),
        "高热": (
            "高热",
            "高烧",
            "持续高烧",
            "高烧不退",
        ),
        "意识改变": (
            "意识改变",
            "意识模糊",
            "意识有点模糊",
            "昏迷",
            "嗜睡明显",
        ),
        "剧烈腹痛": (
            "剧烈腹痛",
            "严重腹痛",
            "持续腹痛",
        ),
        "严重出血": (
            "严重出血",
            "大量出血",
            "止不住血",
            "吐血",
            "便血",
        ),
        "抽搐": (
            "抽搐",
            "惊厥",
            "癫痫发作",
            "全身抽动",
            "抽风",
        ),
        "视力突然下降": (
            "视力突然下降",
            "突然看不清",
            "一只眼看不清",
            "一只眼看不见",
            "眼前黑影",
            "视野缺损",
            "突发视物模糊",
        ),
        "严重过敏反应": (
            "严重过敏反应",
            "严重过敏",
            "过敏反应严重",
            "全身风团",
            "过敏后喘不上气",
            "皮疹伴呼吸困难",
        ),
        "喉头紧缩或面唇肿胀": (
            "喉头紧缩或面唇肿胀",
            "喉咙发紧",
            "喉头紧",
            "喉咙堵",
            "嘴唇肿",
            "眼睑肿",
            "面部肿胀",
            "脸肿",
        ),
        "持续呕吐或腹泻": (
            "持续呕吐或腹泻",
            "持续呕吐",
            "一直吐",
            "频繁呕吐",
            "严重腹泻",
            "一直拉肚子",
            "腹泻不止",
        ),
        "脱水表现": (
            "脱水表现",
            "明显口渴",
            "尿很少",
            "少尿",
            "皮肤干",
            "脱水",
            "眼窝凹陷",
        ),
        "尿血或尿潴留": (
            "尿血或尿潴留",
            "尿血",
            "血尿",
            "完全尿不出来",
            "尿潴留",
            "排不出尿",
        ),
        "孕期出血或腹痛": (
            "孕期出血或腹痛",
            "怀孕出血",
            "孕期出血",
            "孕期腹痛",
            "孕妇腹痛",
            "见红",
            "怀孕后下腹痛",
        ),
        "儿童精神差或反应差": (
            "儿童精神差或反应差",
            "孩子精神差",
            "宝宝精神差",
            "精神差",
            "反应差",
            "嗜睡",
            "叫不醒",
            "奶量明显减少",
            "吃奶差",
        ),
        "严重皮疹或快速扩散": (
            "严重皮疹或快速扩散",
            "皮疹迅速扩散",
            "全身皮疹",
            "皮肤大片红肿",
            "皮疹伴高热",
            "紫癜",
            "皮肤出血点",
        ),
        "药物过量或中毒风险": (
            "药物过量或中毒风险",
            "药物过量",
            "误服药物",
            "吃了很多药",
            "煤气中毒",
            "一氧化碳中毒",
            "农药中毒",
            "酒精中毒",
        ),
    },
    "low_risk_factor": {
        "年龄较轻": (
            "年龄较轻",
            "年轻",
            "二十多岁",
            "大学生",
            "青年",
        ),
        "近期压力大": (
            "近期压力大",
            "压力大",
            "学习压力",
            "工作压力",
            "精神压力",
        ),
        "熬夜或睡眠不足": (
            "熬夜或睡眠不足",
            "熬夜",
            "睡眠不足",
            "没睡好",
            "休息不好",
            "经常熬夜",
        ),
        "无发热": (
            "无发热",
            "没有发烧",
            "没有发热",
            "不发烧",
            "未发热",
            "无发烧",
            "未见发热",
            "没有明显发热",
            "无明显发热",
            "没有明显发烧",
            "无明显发烧",
            "体温正常",
            "不烧",
            "没烧",
            "目前不发烧",
            "目前没有发热",
        ),
        "无咳嗽": (
            "无咳嗽",
            "没有咳嗽",
            "不咳嗽",
            "未咳嗽",
            "无咳",
            "没有明显咳嗽",
            "无明显咳嗽",
            "不怎么咳",
            "很少咳嗽",
            "没怎么咳",
            "目前不咳嗽",
            "目前没有咳嗽",
        ),
        "无呼吸困难": (
            "无呼吸困难",
            "没有呼吸困难",
            "不喘",
            "没有气短",
            "没有喘不上气",
            "呼吸还可以",
        ),
        "无意识改变": (
            "无意识改变",
            "意识清楚",
            "没有意识模糊",
            "反应正常",
            "神志清楚",
        ),
        "症状较轻": (
            "症状较轻",
            "症状轻",
            "不严重",
            "轻微",
            "不是特别疼",
        ),
        "症状短暂且已缓解": (
            "症状短暂且已缓解",
            "很快好了",
            "已经缓解",
            "自己缓解",
            "持续时间很短",
            "一会儿就好了",
        ),
        "无持续加重": (
            "无持续加重",
            "没有加重",
            "没有持续加重",
            "没有越来越严重",
            "症状没有变重",
        ),
        "无明显外伤": (
            "无明显外伤",
            "无外伤",
            "没有外伤",
            "没有受伤",
        ),
        "休息后改善": (
            "休息后改善",
            "休息后缓解",
            "休息会好转",
        ),
    },
    "consult_department": {
        "急诊": (
            "急诊",
            "急诊科",
            "急救",
            "急诊内科",
        ),
        "全科/普通内科": (
            "全科/普通内科",
            "全科",
            "普通内科",
            "内科",
            "社区医院",
            "基层医院",
            "普通内科门诊",
            "综合内科",
            "发热门诊",
            "普通门诊",
            "综合门诊",
        ),
        "心内科": (
            "心内科",
            "心血管内科",
            "心脏科",
            "胸痛中心",
            "心血管科",
        ),
        "神经内科": (
            "神经内科",
            "神经科",
        ),
        "呼吸科": (
            "呼吸科",
            "呼吸内科",
            "咳嗽门诊",
            "呼吸内科门诊",
        ),
        "消化科": (
            "消化科",
            "消化内科",
        ),
        "骨科/康复科": (
            "骨科/康复科",
            "骨科",
            "康复科",
            "疼痛科",
            "运动医学",
        ),
        "心理/精神心理科": (
            "心理/精神心理科",
            "心理",
            "精神心理",
            "精神科",
            "心理科",
        ),
        "泌尿外科/肾内科": (
            "泌尿外科/肾内科",
            "泌尿外科",
            "肾内科",
            "肾脏科",
            "泌尿科",
            "尿路感染门诊",
        ),
        "皮肤科/变态反应科": (
            "皮肤科/变态反应科",
            "皮肤科",
            "过敏科",
            "变态反应科",
            "皮肤过敏科",
        ),
        "耳鼻喉科": (
            "耳鼻喉科",
            "五官科",
            "耳科",
            "鼻科",
            "咽喉科",
        ),
        "眼科": (
            "眼科",
            "眼科急诊",
        ),
        "妇产科": (
            "妇产科",
            "妇科",
            "产科",
            "妇产科",
        ),
        "儿科": (
            "儿科",
            "儿科",
            "小儿内科",
            "儿童医院",
            "儿科急诊",
        ),
        "内分泌科": (
            "内分泌科",
            "内分泌科",
            "糖尿病门诊",
            "甲状腺门诊",
        ),
        "血液科/风湿免疫科": (
            "血液科/风湿免疫科",
            "血液科",
            "风湿免疫科",
            "免疫科",
            "风湿科",
        ),
        "不确定": (
            "不确定",
            "无法判断",
            "不好判断",
        ),
    },
}

_PUNCT_RE = re.compile(r"[，。；：！？、,.!?:;/\\|（）()\[\]{}<>《》“”\"'‘’`~·—\-]+")
_SPACE_RE = re.compile(r"\s+")
_CLAUSE_SPLIT_RE = re.compile(r"[。！？!?；;：:\n\r]+|(?:，|,)|(?:但是|但|不过|然而)")
_MAJOR_SEGMENT_SPLIT_RE = re.compile(r"[。！？!?；;\n\r]+|(?:但是|但|不过|然而)")
_NEGATION_MARKERS = (
    "无",
    "没有",
    "未",
    "未见",
    "不伴",
    "否认",
    "暂无",
    "未发现",
    "无明显",
    "不考虑",
    "不太像",
    "暂不支持",
    "证据不足",
    "不需要",
)
_NEGATING_EXCLUSION_PATTERNS = (
    "基本排除{alias}",
    "可以排除{alias}",
    "已排除{alias}",
    "{alias}可能性低",
    "{alias}可能性较低",
    "{alias}问题可能性低",
    "{alias}问题可能性较低",
    "不太像{alias}",
    "暂不支持{alias}",
    "目前不考虑{alias}",
    "不考虑{alias}",
    "证据不足以支持{alias}",
    "证据不足支持{alias}",
)
_NON_NEGATING_EXCLUSION_PATTERNS = (
    "不能排除{alias}",
    "无法排除{alias}",
    "不排除{alias}",
    "需要排除{alias}",
    "需排除{alias}",
    "建议排除{alias}",
    "仍需排除{alias}",
)
_CONDITIONAL_MARKERS = (
    "如果",
    "若",
    "一旦",
    "如果出现",
    "若出现",
    "如出现",
    "一旦出现",
    "当出现",
    "如果伴随",
    "若伴随",
    "如伴随",
    "当伴随",
)
_CONDITIONAL_RE = re.compile(
    r"((^|[，,；;])\s*(如果|若|如|一旦|当)[^。；，,]*(应|则|需要|建议)|出现[^。；，,]*则|伴随[^。；，,]*时|(^|[，,；;])\s*(如果|若|如|一旦|当)[^。；，,]*(出现|伴随))"
)
_LEADING_RU_CONDITIONAL_RE = re.compile(r"(^|[，,；;。])\s*如(?=[^，,；;。]{0,20}(出现|伴随|加重|持续|胸痛|气短|意识|高热|腹痛|出血|呕吐|腹泻|尿血|见红|反应差|叫不醒))")
_AGE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*岁")
_TEMPERATURE_WITH_CONTEXT_RE = re.compile(
    r"(?:体温|发烧到|发热到|发热|发烧|烧到|高于|超过)\s*(\d{2}(?:\.\d)?)\s*(?:°c|°f|℃|度|c)?(?:多)?",
    re.IGNORECASE,
)
_TEMPERATURE_WITH_UNIT_RE = re.compile(r"(?<!\d)(\d{2}(?:\.\d)?)\s*(?:°c|℃|度|c)(?:多)?(?!岁)", re.IGNORECASE)
_STRONG_NEURO_SIGNS = {"突发无力或言语不清", "意识改变", "严重头痛", "抽搐"}
_STRONG_BLEEDING_SIGNS = {"严重出血"}
_STRONG_RESP_SIGNS = {"气短"}
_STRONG_INFECTION_SIGNS = {"高热", "意识改变"}
_STRONG_ABDOMINAL_SIGNS = {"剧烈腹痛"}
_STRONG_ALLERGY_SIGNS = {"严重过敏反应", "喉头紧缩或面唇肿胀"}
_STRONG_VISION_SIGNS = {"视力突然下降"}
_STRONG_PREGNANCY_SIGNS = {"孕期出血或腹痛"}
_STRONG_PEDIATRIC_SIGNS = {"儿童精神差或反应差", "抽搐"}
_STRONG_POISONING_SIGNS = {"药物过量或中毒风险"}
_STRONG_URINARY_SIGNS = {"尿血或尿潴留"}
_GI_FLUID_RISK_SIGNS = {"持续呕吐或腹泻", "脱水表现"}
_SKIN_RISK_SIGNS = {"严重皮疹或快速扩散"}


def normalize_text_for_match(text: Any) -> str:
    """Normalize text into a compact form for deterministic matching."""
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).strip().lower()
    if not normalized:
        return ""
    normalized = _PUNCT_RE.sub(" ", normalized)
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    return normalized.replace(" ", "")


def split_clauses(text: Any) -> list[Clause]:
    """Split free text into short clauses while keeping both raw and normalized forms."""
    if text is None:
        return []
    raw_text = unicodedata.normalize("NFKC", str(text)).strip()
    if not raw_text:
        return []
    parts = [part.strip() for part in _CLAUSE_SPLIT_RE.split(raw_text) if part and part.strip()]
    clauses = [Clause(raw=part, normalized=normalize_text_for_match(part)) for part in parts]
    return [clause for clause in clauses if clause.normalized]


def flatten_raw_value(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, Mapping):
        flattened: list[str] = []
        for value in raw_value.values():
            flattened.extend(flatten_raw_value(value))
        return flattened
    if isinstance(raw_value, (list, tuple)):
        flattened = []
        for item in raw_value:
            flattened.extend(flatten_raw_value(item))
        return flattened
    if isinstance(raw_value, set):
        flattened = []
        for item in sorted(raw_value, key=lambda value: str(value)):
            flattened.extend(flatten_raw_value(item))
        return flattened
    return [str(raw_value)]


def get_medical_objects() -> list[dict[str, Any]]:
    return [
        {
            "object_id": schema.object_id,
            "label": schema.label,
            "mode": schema.mode,
            "options": list(schema.options),
        }
        for schema in MEDICAL_OBJECTS
    ]


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _order_by_schema(object_id: str, matched: Iterable[str]) -> list[str]:
    schema = OBJECT_INDEX[object_id]
    matched_set = set(matched)
    return [option for option in schema.options if option in matched_set]


TRUNCATION_PRIORITY: dict[str, tuple[str, ...]] = {
    "risk_signal": (
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
    ),
    "possible_cause": (
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
    ),
    "consult_department": (
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
    ),
}


def _rank_facts_for_truncation(object_id: str, facts: Sequence[str]) -> list[str]:
    ordered = _order_by_schema(object_id, facts)
    priority = TRUNCATION_PRIORITY.get(object_id)
    if not priority:
        return ordered
    rank_map = {fact: idx for idx, fact in enumerate(priority)}
    return sorted(
        ordered,
        key=lambda fact: (rank_map.get(fact, len(rank_map)), OBJECT_INDEX[object_id].options.index(fact)),
    )


def _truncate_candidates(
    object_id: str,
    facts: list[str],
    max_candidates_per_object: int | None,
) -> list[str]:
    if max_candidates_per_object is None:
        return facts
    if max_candidates_per_object <= 0:
        return []
    ranked = _rank_facts_for_truncation(object_id, facts)
    kept = ranked[:max_candidates_per_object]
    return _order_by_schema(object_id, kept)


def _build_normalized_alias_index() -> dict[str, dict[str, tuple[str, ...]]]:
    index: dict[str, dict[str, tuple[str, ...]]] = {}
    for object_id, option_map in OBJECT_ALIASES.items():
        index[object_id] = {}
        for option, aliases in option_map.items():
            normalized_aliases = _dedupe_keep_order(
                normalize_text_for_match(alias) for alias in (option, *aliases)
            )
            index[object_id][option] = tuple(alias for alias in normalized_aliases if alias)
    return index


NORMALIZED_ALIAS_INDEX = _build_normalized_alias_index()


def _build_clause_pool(texts: Sequence[str], *, include_splits: bool = True) -> list[Clause]:
    clauses: list[Clause] = []
    seen: set[tuple[str, str]] = set()
    for text in texts:
        if not text:
            continue
        raw_text = unicodedata.normalize("NFKC", str(text)).strip()
        if not raw_text:
            continue
        full_clause = Clause(raw=raw_text, normalized=normalize_text_for_match(raw_text))
        key = (full_clause.raw, full_clause.normalized)
        if full_clause.normalized and key not in seen:
            clauses.append(full_clause)
            seen.add(key)
        if include_splits:
            for clause in split_clauses(raw_text):
                key = (clause.raw, clause.normalized)
                if key not in seen:
                    clauses.append(clause)
                    seen.add(key)
    return clauses


def _find_occurrences(text: str, alias: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while alias:
        index = text.find(alias, start)
        if index < 0:
            break
        positions.append(index)
        start = index + len(alias)
    return positions


def _normalized_prefix_length(text: str, raw_index: int) -> int:
    return len(normalize_text_for_match(text[:raw_index]))


def _find_conditional_start_positions(raw_text: str) -> list[int]:
    raw_text = unicodedata.normalize("NFKC", raw_text)
    if not raw_text or not normalize_text_for_match(raw_text):
        return []

    positions: list[int] = []
    for match in re.finditer(r"如果|若|一旦|如果出现|若出现|如出现|一旦出现|当出现|如果伴随|若伴随|如伴随|当伴随", raw_text):
        positions.append(_normalized_prefix_length(raw_text, match.start()))

    for match in _LEADING_RU_CONDITIONAL_RE.finditer(raw_text):
        positions.append(_normalized_prefix_length(raw_text, match.start()))
    for match in re.finditer(r"出现[^。；，,]*则", raw_text):
        positions.append(_normalized_prefix_length(raw_text, match.start()))
    for match in re.finditer(r"伴随[^。；，,]*时", raw_text):
        positions.append(_normalized_prefix_length(raw_text, match.start()))

    return sorted(set(position for position in positions if position >= 0))


def _strip_conditional_tail_for_current_judgment(text: str) -> str:
    raw_text = unicodedata.normalize("NFKC", str(text)).strip()
    if not raw_text or not any(hint in raw_text for hint in _CURRENT_JUDGMENT_HINTS):
        return raw_text

    condition_starts = _find_conditional_start_positions(raw_text)
    if not condition_starts:
        return raw_text

    first_current_start = min(raw_text.find(hint) for hint in _CURRENT_JUDGMENT_HINTS if hint in raw_text)
    first_condition_start = min(condition_starts)
    if first_condition_start <= _normalized_prefix_length(raw_text, first_current_start):
        return raw_text

    kept = raw_text
    for raw_marker in ("如果", "若", "一旦", "当出现", "当伴随"):
        position = raw_text.find(raw_marker, first_current_start)
        if position > first_current_start:
            kept = raw_text[:position]
            break
    else:
        leading_ru_match = _LEADING_RU_CONDITIONAL_RE.search(raw_text)
        if leading_ru_match and _normalized_prefix_length(raw_text, leading_ru_match.start()) == first_condition_start:
            kept = raw_text[: leading_ru_match.start()]
        else:
            for match in re.finditer(r"出现[^。；，,]*则|伴随[^。；，,]*时", raw_text):
                if _normalized_prefix_length(raw_text, match.start()) == first_condition_start:
                    kept = raw_text[: match.start()]
                    break
            else:
                return raw_text

    return kept.rstrip("，,；;。 ")


def _extract_temperature_values(texts: Sequence[str]) -> list[float]:
    values: list[float] = []
    for text in texts:
        raw_text = unicodedata.normalize("NFKC", str(text))
        for match in _TEMPERATURE_WITH_CONTEXT_RE.finditer(raw_text):
            values.append(float(match.group(1)))
        for match in _TEMPERATURE_WITH_UNIT_RE.finditer(raw_text):
            values.append(float(match.group(1)))
    return values


def _detect_high_fever(texts: Sequence[str]) -> bool:
    return any(value >= 39.0 for value in _extract_temperature_values(texts))


def _detect_moderate_fever(texts: Sequence[str]) -> bool:
    return any(38.0 <= value < 39.0 for value in _extract_temperature_values(texts))


def _matches_alias_templates(clause: Clause, alias: str, templates: Sequence[str]) -> bool:
    alias_norm = normalize_text_for_match(alias)
    if not alias_norm:
        return False
    clause_text = clause.normalized
    return any(normalize_text_for_match(template.format(alias=alias_norm)) in clause_text for template in templates)


def is_true_exclusion_context(clause: Clause, alias: str) -> bool:
    alias_norm = normalize_text_for_match(alias)
    if not alias_norm:
        return False
    clause_text = clause.normalized
    if _matches_alias_templates(clause, alias, _NEGATING_EXCLUSION_PATTERNS):
        return True
    return bool(
        re.search(
            rf"(基本排除|可以排除|已排除){re.escape(alias_norm)}|"
            rf"{re.escape(alias_norm)}.{{0,4}}(可能性低|可能性较低)|"
            rf"(不太像|暂不支持|(目前)?不考虑){re.escape(alias_norm)}|"
            rf"证据不足(以)?支持{re.escape(alias_norm)}",
            clause_text,
        )
    )


def is_non_negating_exclusion_context(clause: Clause, alias: str) -> bool:
    alias_norm = normalize_text_for_match(alias)
    if not alias_norm:
        return False
    clause_text = clause.normalized
    if _matches_alias_templates(clause, alias, _NON_NEGATING_EXCLUSION_PATTERNS):
        return True
    return bool(
        re.search(
            rf"(不能排除|无法排除|不排除|需要排除|需排除|建议排除|仍需排除){re.escape(alias_norm)}",
            clause_text,
        )
    )


def is_negated_context(clause: Clause, alias: str) -> bool:
    alias_norm = normalize_text_for_match(alias)
    if not alias_norm or alias_norm not in clause.normalized:
        return False
    if is_non_negating_exclusion_context(clause, alias):
        return False
    if is_true_exclusion_context(clause, alias):
        return True

    for position in _find_occurrences(clause.normalized, alias_norm):
        prefix = clause.normalized[max(0, position - 12) : position]
        if any(prefix.endswith(marker) for marker in _NEGATION_MARKERS):
            return True
        for marker in _NEGATION_MARKERS:
            idx = prefix.rfind(marker)
            if idx >= 0:
                tail = prefix[idx + len(marker) :]
                if len(tail) <= 8 and "但" not in tail and "不过" not in tail:
                    return True
    return False


def is_conditional_context(clause: Clause, alias: str) -> bool:
    alias_norm = normalize_text_for_match(alias)
    if not alias_norm or alias_norm not in clause.normalized:
        return False
    raw = unicodedata.normalize("NFKC", clause.raw)
    normalized_raw = normalize_text_for_match(raw)
    alias_positions = _find_occurrences(normalized_raw, alias_norm)
    if any(raw.startswith(hint) for hint in _CURRENT_JUDGMENT_HINTS):
        stripped_current = _strip_conditional_tail_for_current_judgment(raw)
        if normalize_text_for_match(stripped_current) == normalized_raw:
            return False
    conditional_positions = _find_conditional_start_positions(raw)

    if conditional_positions:
        condition_start = min(conditional_positions)
        if any(position < condition_start for position in alias_positions):
            return False
        if any(position >= condition_start for position in alias_positions):
            return True
    if any(
        normalized_raw.startswith(normalize_text_for_match(marker))
        for marker in ("如果", "若", "一旦")
    ):
        return True
    if _LEADING_RU_CONDITIONAL_RE.match(raw):
        return True
    regex_hit = bool(_CONDITIONAL_RE.search(raw))
    return regex_hit


def _alias_positive_in_clause(
    clause: Clause,
    alias: str,
    *,
    allow_conditional: bool,
) -> bool:
    alias_norm = normalize_text_for_match(alias)
    if not alias_norm or alias_norm not in clause.normalized:
        return False
    if is_negated_context(clause, alias):
        return False
    if not allow_conditional and is_conditional_context(clause, alias):
        return False
    return True


def _match_object_facts(
    object_id: str,
    texts: Sequence[str],
    *,
    allow_conditional: bool = False,
) -> list[str]:
    clauses = _build_clause_pool(
        texts,
        include_splits=object_id not in {"danger_signal", "urgency_level"},
    )
    if not clauses:
        return []

    matched: list[str] = []
    option_aliases = NORMALIZED_ALIAS_INDEX[object_id]

    for option in OBJECT_INDEX[object_id].options:
        for clause in clauses:
            if any(
                _alias_positive_in_clause(clause, alias, allow_conditional=allow_conditional)
                for alias in option_aliases[option]
            ):
                matched.append(option)
                break
    return _order_by_schema(object_id, matched)


def _detect_young_age(texts: Sequence[str]) -> bool:
    for text in texts:
        raw = unicodedata.normalize("NFKC", str(text))
        for match in _AGE_RE.finditer(raw):
            if int(match.group(1)) <= 35:
                return True
        normalized = normalize_text_for_match(raw)
        if any(keyword in normalized for keyword in ("二十多岁", "大学生", "青年", "年轻")):
            return True
    return False


def _apply_low_risk_combo_cases(facts: list[str], texts: Sequence[str]) -> list[str]:
    result = list(facts)
    combo_re = re.compile(
        r"(没有|无|没|不)(发烧|发热)(咳嗽|和咳嗽)|"
        r"(没有|无|没|不)(发烧|发热)(也)?(没有|无|没|不)咳嗽"
    )
    for text in texts:
        for clause in split_clauses(text):
            if combo_re.search(clause.raw):
                result.extend(["无发热", "无咳嗽"])
                break
    return _order_by_schema("low_risk_factor", _dedupe_keep_order(result))


def _apply_low_risk_special_cases(facts: list[str], texts: Sequence[str]) -> list[str]:
    result = _apply_low_risk_combo_cases(facts, texts)
    if _detect_young_age(texts):
        result.append("年龄较轻")
    return _order_by_schema("low_risk_factor", _dedupe_keep_order(result))


def _apply_risk_signal_special_cases(facts: list[str], texts: Sequence[str]) -> list[str]:
    result = list(facts)
    if _detect_high_fever(texts):
        result.append("高热")
    return _order_by_schema("risk_signal", _dedupe_keep_order(result))


def _apply_possible_cause_special_cases(facts: list[str], texts: Sequence[str]) -> list[str]:
    result = list(facts)
    if _detect_high_fever(texts) or _detect_moderate_fever(texts):
        result.append("感染相关")
    return _cleanup_multi_facts("possible_cause", result)


def _cleanup_multi_facts(object_id: str, facts: list[str]) -> list[str]:
    ordered = _order_by_schema(object_id, _dedupe_keep_order(facts))
    if object_id == "possible_cause" and len(ordered) > 1 and "原因不明" in ordered:
        ordered = [fact for fact in ordered if fact != "原因不明"]
    if object_id == "consult_department" and len(ordered) > 1 and "不确定" in ordered:
        ordered = [fact for fact in ordered if fact != "不确定"]
    return ordered


def _normalize_medical_fact_detailed(
    object_id: str,
    raw_value: Any,
    *,
    user_text: str = "",
) -> tuple[list[str], bool]:
    schema = OBJECT_INDEX[object_id]
    texts = flatten_raw_value(raw_value)

    if not texts and object_id in {"risk_signal", "low_risk_factor"} and user_text.strip():
        texts = [user_text]

    if object_id == "urgency_level":
        matched = _match_urgency_level_with_context_priority(texts)
        if not matched:
            matched = _match_object_facts(object_id, texts, allow_conditional=False)
    else:
        matched = _match_object_facts(object_id, texts, allow_conditional=False)

    if object_id == "low_risk_factor":
        matched = _apply_low_risk_special_cases(matched, texts or ([user_text] if user_text.strip() else []))
    elif object_id == "risk_signal":
        matched = _apply_risk_signal_special_cases(matched, texts)
    elif object_id == "possible_cause":
        matched = _apply_possible_cause_special_cases(matched, texts)
    elif object_id == "consult_department":
        matched = _cleanup_multi_facts(object_id, matched)

    if matched:
        if schema.mode == "single":
            return [matched[0]], False
        return matched, False

    if schema.fallback:
        return list(schema.fallback), True
    return [], False


def normalize_medical_fact(object_id: str, raw_value: Any, *, user_text: str = "") -> list[str]:
    """
    Normalize one object's raw model field into standard facts.

    The optional user_text parameter is only a limited fallback for a few
    observable fields when raw_value is missing. It is not the full user-text
    patch pipeline. Full user_text extraction and safety patching should go
    through _extract_user_text_patch_facts() or normalize_model_medical_output().
    """
    if object_id not in OBJECT_INDEX:
        raise ValueError(f"Unknown medical object_id: {object_id}")
    facts, _used_fallback = _normalize_medical_fact_detailed(
        object_id,
        raw_value,
        user_text=user_text,
    )
    return facts


def _update_single_select(current: list[str], target: str, object_id: str) -> list[str]:
    schema = OBJECT_INDEX[object_id]
    current_value = current[0] if current else schema.options[-1]
    if schema.options.index(target) < schema.options.index(current_value):
        return [target]
    return current or [current_value]


def _add_multi_fact(current: list[str], object_id: str, fact: str) -> list[str]:
    return _cleanup_multi_facts(object_id, [*current, fact])


def _collect_text_corpus(
    model_output: Any,
    raw_structured_analysis: Mapping[str, Any],
    user_explanation: str,
) -> list[str]:
    texts: list[str] = []
    if isinstance(model_output, Mapping):
        for key, value in model_output.items():
            if key in {"structured_analysis", "user_explanation"}:
                continue
            texts.extend(flatten_raw_value(value))
    else:
        texts.extend(flatten_raw_value(model_output))
    texts.extend(flatten_raw_value(raw_structured_analysis))
    if user_explanation:
        texts.append(user_explanation)
    return _dedupe_keep_order(texts)


def _empty_patch_map() -> dict[str, list[str]]:
    return {schema.object_id: [] for schema in MEDICAL_OBJECTS}


def _extract_user_text_patch_facts(user_text: str) -> dict[str, list[str]]:
    patches = _empty_patch_map()
    if not user_text.strip():
        return patches
    texts = [user_text]
    for schema in MEDICAL_OBJECTS:
        object_id = schema.object_id
        patches[object_id] = _facts_from_text_corpus(object_id, texts)
    return patches


def _merge_patch_map_into_normalized(
    normalized: dict[str, list[str]],
    patch_map: Mapping[str, Sequence[str]],
) -> None:
    for schema in MEDICAL_OBJECTS:
        facts = list(patch_map.get(schema.object_id, []))
        if not facts:
            continue
        if schema.mode == "single":
            normalized[schema.object_id] = _update_single_select(
                normalized.get(schema.object_id, []),
                facts[0],
                schema.object_id,
            )
        else:
            normalized[schema.object_id] = _cleanup_multi_facts(
                schema.object_id,
                [*normalized.get(schema.object_id, []), *facts],
            )


def _facts_from_text_corpus(object_id: str, texts: Sequence[str]) -> list[str]:
    if object_id == "urgency_level":
        facts = _match_urgency_level_with_context_priority(texts)
        if not facts:
            facts = _match_object_facts(object_id, texts, allow_conditional=False)
    else:
        facts = _match_object_facts(object_id, texts, allow_conditional=False)
    if object_id == "low_risk_factor":
        return _apply_low_risk_special_cases(facts, texts)
    if object_id == "risk_signal":
        return _apply_risk_signal_special_cases(facts, texts)
    if object_id == "possible_cause":
        return _apply_possible_cause_special_cases(facts, texts)
    if object_id == "consult_department":
        return _cleanup_multi_facts(object_id, facts)
    return facts


def _empty_field_status() -> dict[str, dict[str, bool]]:
    return {
        schema.object_id: {
            "missing": False,
            "used_fallback": False,
        }
        for schema in MEDICAL_OBJECTS
    }


_CURRENT_JUDGMENT_HINTS = (
    "目前",
    "当前",
    "现在",
    "现阶段",
    "暂时",
    "结合目前情况",
)


def _match_urgency_level_with_context_priority(texts: Sequence[str]) -> list[str]:
    current_snippets: list[str] = []
    current_clauses: list[str] = []
    nonconditional_segments: list[str] = []

    for text in texts:
        raw_text = unicodedata.normalize("NFKC", str(text)).strip()
        if not raw_text:
            continue
        for hint in _CURRENT_JUDGMENT_HINTS:
            current_snippets.extend(
                _strip_conditional_tail_for_current_judgment(match.group(0).strip())
                for match in re.finditer(
                    rf"{re.escape(hint)}[^。！？!?；;\n\r]*",
                    raw_text,
                )
            )
        for clause in split_clauses(raw_text):
            if any(hint in clause.raw for hint in _CURRENT_JUDGMENT_HINTS):
                current_clauses.append(_strip_conditional_tail_for_current_judgment(clause.raw))
        for segment in _MAJOR_SEGMENT_SPLIT_RE.split(raw_text):
            segment = segment.strip()
            if not segment:
                continue
            clause = Clause(raw=segment, normalized=normalize_text_for_match(segment))
            if not is_conditional_context(clause, clause.raw):
                nonconditional_segments.append(segment)

    if current_snippets:
        facts = _match_object_facts("urgency_level", current_snippets, allow_conditional=False)
        if facts:
            return facts

    if current_clauses:
        facts = _match_object_facts("urgency_level", current_clauses, allow_conditional=False)
        if facts:
            return facts

    if nonconditional_segments:
        facts = _match_object_facts("urgency_level", nonconditional_segments, allow_conditional=False)
        if facts:
            return facts

    return []


def _apply_safety_overrides(
    normalized: dict[str, list[str]],
    *,
    text_corpus: Sequence[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    warnings: list[str] = []
    overrides: list[dict[str, Any]] = []
    observed_risk_signals = _facts_from_text_corpus("risk_signal", text_corpus)
    normalized["risk_signal"] = _cleanup_multi_facts(
        "risk_signal",
        [*normalized.get("risk_signal", []), *observed_risk_signals],
    )
    risk_set = set(normalized["risk_signal"])

    def record_override(rule: str, affected_objects: Sequence[str], message: str) -> None:
        warnings.append(message)
        overrides.append(
            {
                "rule": rule,
                "affected_objects": list(affected_objects),
                "message": message,
            }
        )

    pediatric_hint = _text_has_pediatric_hint(text_corpus)
    pregnancy_hint = _text_has_pregnancy_hint(text_corpus)
    allergy_phrase = _text_has_any_nonconditional(
        text_corpus,
        (
            "过敏后喘不上气",
            "嘴唇肿并呼吸困难",
            "喉咙发紧喘不上气",
            "皮疹伴呼吸困难",
            "全身风团喘不上气",
        ),
    )
    vision_phrase = _text_has_any_nonconditional(
        text_corpus,
        ("突然看不清", "一只眼看不见", "视力突然下降", "视野缺损", "眼前黑影"),
    )
    poisoning_phrase = _text_has_any_nonconditional(
        text_corpus,
        ("药物过量", "误服药物", "吃了很多药", "吃错药", "煤气中毒", "一氧化碳中毒", "农药中毒", "酒精中毒", "中毒"),
    )
    urinary_phrase = _text_has_any_nonconditional(
        text_corpus,
        ("尿血", "血尿", "排不出尿", "完全尿不出来", "尿潴留"),
    )
    pregnancy_bleeding_or_pain = pregnancy_hint and _text_has_any_nonconditional(
        text_corpus,
        ("出血", "见红", "腹痛", "下腹痛"),
    )
    urinary_fever_combo = _text_has_any_nonconditional(text_corpus, ("腰痛",)) and "高热" in risk_set

    chest_support = {"胸闷", "胸痛"} & risk_set
    chest_escalators = {
        "活动后加重",
        "休息后缓解",
        "左肩或左臂不适",
        "气短",
        "出汗",
        "恶心",
        "头晕",
        "心跳快或心慌",
    } & risk_set
    strong_chest_phrase = _contains_any_nonconditional_phrase(
        text_corpus,
        (
            "胸痛持续不缓解",
            "胸闷持续不缓解",
            "胸痛明显加重",
            "胸闷明显加重",
            "胸部压迫感持续不缓解",
        ),
    )
    if chest_support and chest_escalators:
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "可能存在危险信号",
            "danger_signal",
        )
        normalized["possible_cause"] = _add_multi_fact(
            normalized.get("possible_cause", []),
            "possible_cause",
            "心血管相关",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "尽快线下就医",
            "urgency_level",
        )
        record_override(
            "chest-risk",
            ("danger_signal", "urgency_level", "possible_cause"),
            "Applied chest-risk safety override",
        )
    if chest_support and (
        strong_chest_phrase or ("胸痛" in risk_set and {"气短", "出汗", "恶心", "头晕"} & risk_set)
    ):
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "存在明显危险信号",
            "danger_signal",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "立即急诊",
            "urgency_level",
        )
        normalized["possible_cause"] = _add_multi_fact(
            normalized.get("possible_cause", []),
            "possible_cause",
            "心血管相关",
        )
        normalized["consult_department"] = _add_multi_fact(
            normalized.get("consult_department", []),
            "consult_department",
            "急诊",
        )
        record_override(
            "strong-chest-risk",
            ("danger_signal", "urgency_level", "possible_cause", "consult_department"),
            "Applied strong chest-emergency safety override",
        )

    if _STRONG_NEURO_SIGNS & risk_set:
        normalized["possible_cause"] = _add_multi_fact(
            normalized.get("possible_cause", []),
            "possible_cause",
            "神经系统相关",
        )
        normalized["consult_department"] = _add_multi_fact(
            normalized.get("consult_department", []),
            "consult_department",
            "神经内科",
        )
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "存在明显危险信号",
            "danger_signal",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "立即急诊",
            "urgency_level",
        )
        normalized["consult_department"] = _add_multi_fact(
            normalized.get("consult_department", []),
            "consult_department",
            "急诊",
        )
        record_override(
            "neuro-risk",
            ("danger_signal", "urgency_level", "possible_cause", "consult_department"),
            "Applied neuro-emergency safety override",
        )

    if _STRONG_RESP_SIGNS & risk_set:
        normalized["possible_cause"] = _add_multi_fact(
            normalized.get("possible_cause", []),
            "possible_cause",
            "呼吸系统相关",
        )
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "可能存在危险信号",
            "danger_signal",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "尽快线下就医",
            "urgency_level",
        )
        if _contains_any_nonconditional_phrase(
            text_corpus,
            ("呼吸困难", "喘不上气", "明显气短", "喘不过气", "呼吸费劲"),
        ):
            normalized["urgency_level"] = _update_single_select(
                normalized.get("urgency_level", []),
                "立即急诊",
                "urgency_level",
            )
            normalized["consult_department"] = _add_multi_fact(
                normalized.get("consult_department", []),
                "consult_department",
                "急诊",
            )
        record_override(
            "respiratory-risk",
            ("danger_signal", "urgency_level", "possible_cause"),
            "Applied respiratory-risk safety override",
        )

    severe_allergy_risk = (
        "严重过敏反应" in risk_set
        or ("喉头紧缩或面唇肿胀" in risk_set and bool({"气短", "严重过敏反应"} & risk_set))
        or allergy_phrase
    )
    if severe_allergy_risk:
        normalized["possible_cause"] = _add_multi_fact(
            normalized.get("possible_cause", []),
            "possible_cause",
            "皮肤或过敏相关",
        )
        normalized["consult_department"] = _add_multi_fact(
            normalized.get("consult_department", []),
            "consult_department",
            "皮肤科/变态反应科",
        )
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "存在明显危险信号",
            "danger_signal",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "立即急诊",
            "urgency_level",
        )
        normalized["consult_department"] = _add_multi_fact(
            normalized.get("consult_department", []),
            "consult_department",
            "急诊",
        )
        record_override(
            "severe-allergy-risk",
            ("danger_signal", "urgency_level", "possible_cause", "consult_department"),
            "Applied severe-allergy emergency safety override",
        )

    has_high_fever = "高热" in risk_set
    infection_emergency = has_high_fever and bool({"意识改变", "气短"} & risk_set)
    if has_high_fever:
        normalized["possible_cause"] = _add_multi_fact(
            normalized.get("possible_cause", []),
            "possible_cause",
            "感染相关",
        )
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "可能存在危险信号",
            "danger_signal",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "尽快线下就医",
            "urgency_level",
        )
        if infection_emergency:
            normalized["danger_signal"] = _update_single_select(
                normalized.get("danger_signal", []),
                "存在明显危险信号",
                "danger_signal",
            )
            normalized["urgency_level"] = _update_single_select(
                normalized.get("urgency_level", []),
                "立即急诊",
                "urgency_level",
            )
            normalized["consult_department"] = _add_multi_fact(
                normalized.get("consult_department", []),
                "consult_department",
                "急诊",
            )
            record_override(
                "infection-with-consciousness-risk",
                ("danger_signal", "urgency_level", "possible_cause", "consult_department"),
                "Applied infection-with-consciousness safety override",
            )
        else:
            record_override(
                "infection-risk",
                ("danger_signal", "urgency_level", "possible_cause"),
                "Applied infection-risk safety override",
            )

    poisoning_risk = bool(_STRONG_POISONING_SIGNS & risk_set) or poisoning_phrase
    if poisoning_risk:
        normalized["possible_cause"] = _add_multi_fact(
            normalized.get("possible_cause", []),
            "possible_cause",
            "药物或中毒相关",
        )
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "存在明显危险信号",
            "danger_signal",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "立即急诊",
            "urgency_level",
        )
        normalized["consult_department"] = _add_multi_fact(
            normalized.get("consult_department", []),
            "consult_department",
            "急诊",
        )
        record_override(
            "poisoning-risk",
            ("danger_signal", "urgency_level", "possible_cause", "consult_department"),
            "Applied poisoning safety override",
        )

    pregnancy_risk = bool(_STRONG_PREGNANCY_SIGNS & risk_set) or pregnancy_bleeding_or_pain
    if pregnancy_risk:
        normalized["possible_cause"] = _add_multi_fact(
            normalized.get("possible_cause", []),
            "possible_cause",
            "妇产/生殖相关",
        )
        normalized["consult_department"] = _add_multi_fact(
            normalized.get("consult_department", []),
            "consult_department",
            "妇产科",
        )
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "可能存在危险信号",
            "danger_signal",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "尽快线下就医",
            "urgency_level",
        )
        pregnancy_emergency = bool({"严重出血", "剧烈腹痛", "头晕", "意识改变"} & risk_set) or _text_has_any_nonconditional(text_corpus, ("大量出血", "晕厥"))
        if pregnancy_emergency:
            normalized["danger_signal"] = _update_single_select(
                normalized.get("danger_signal", []),
                "存在明显危险信号",
                "danger_signal",
            )
            normalized["urgency_level"] = _update_single_select(
                normalized.get("urgency_level", []),
                "立即急诊",
                "urgency_level",
            )
            normalized["consult_department"] = _add_multi_fact(
                normalized.get("consult_department", []),
                "consult_department",
                "急诊",
            )
        record_override(
            "pregnancy-risk",
            ("danger_signal", "urgency_level", "possible_cause", "consult_department"),
            "Applied pregnancy-related safety override",
        )

    pediatric_high_risk = pediatric_hint and bool(
        {"高热", "抽搐", "意识改变", "儿童精神差或反应差", "严重出血", "持续呕吐或腹泻", "脱水表现"} & risk_set
    )
    if pediatric_high_risk:
        normalized["consult_department"] = _add_multi_fact(
            normalized.get("consult_department", []),
            "consult_department",
            "儿科",
        )
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "存在明显危险信号",
            "danger_signal",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "立即急诊",
            "urgency_level",
        )
        normalized["consult_department"] = _add_multi_fact(
            normalized.get("consult_department", []),
            "consult_department",
            "急诊",
        )
        record_override(
            "pediatric-risk",
            ("danger_signal", "urgency_level", "consult_department"),
            "Applied pediatric safety override",
        )

    if _STRONG_ABDOMINAL_SIGNS & risk_set or (
        "恶心" in risk_set and _contains_any_nonconditional_phrase(text_corpus, ("腹痛", "剧烈腹痛", "严重腹痛", "持续腹痛", "便血"))
    ):
        normalized["possible_cause"] = _add_multi_fact(
            normalized.get("possible_cause", []),
            "possible_cause",
            "消化系统相关",
        )
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "可能存在危险信号",
            "danger_signal",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "尽快线下就医",
            "urgency_level",
        )
        record_override(
            "abdominal-risk",
            ("danger_signal", "urgency_level", "possible_cause"),
            "Applied abdominal-pain safety override",
        )

    if _GI_FLUID_RISK_SIGNS & risk_set:
        normalized["possible_cause"] = _add_multi_fact(
            normalized.get("possible_cause", []),
            "possible_cause",
            "消化系统相关",
        )
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "可能存在危险信号",
            "danger_signal",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "尽快线下就医",
            "urgency_level",
        )
        record_override(
            "fluid-loss-risk",
            ("danger_signal", "urgency_level", "possible_cause"),
            "Applied vomiting-diarrhea safety override",
        )
    
    urinary_red_flag = bool(_STRONG_URINARY_SIGNS & risk_set) or urinary_phrase or urinary_fever_combo
    if urinary_red_flag:
        normalized["possible_cause"] = _add_multi_fact(
            normalized.get("possible_cause", []),
            "possible_cause",
            "泌尿系统相关",
        )
        if "高热" in risk_set:
            normalized["possible_cause"] = _add_multi_fact(
                normalized.get("possible_cause", []),
                "possible_cause",
                "感染相关",
            )
        normalized["consult_department"] = _add_multi_fact(
            normalized.get("consult_department", []),
            "consult_department",
            "泌尿外科/肾内科",
        )
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "可能存在危险信号",
            "danger_signal",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "尽快线下就医",
            "urgency_level",
        )
        urinary_emergency = bool({"高热", "意识改变", "严重出血"} & risk_set) or _text_has_any_nonconditional(text_corpus, ("明显气短",))
        if urinary_emergency:
            normalized["danger_signal"] = _update_single_select(
                normalized.get("danger_signal", []),
                "存在明显危险信号",
                "danger_signal",
            )
            normalized["urgency_level"] = _update_single_select(
                normalized.get("urgency_level", []),
                "立即急诊",
                "urgency_level",
            )
            normalized["consult_department"] = _add_multi_fact(
                normalized.get("consult_department", []),
                "consult_department",
                "急诊",
            )
        record_override(
            "urinary-red-flag",
            ("danger_signal", "urgency_level", "possible_cause", "consult_department"),
            "Applied urinary red-flag safety override",
        )

    vision_loss_risk = bool(_STRONG_VISION_SIGNS & risk_set) or vision_phrase
    if vision_loss_risk:
        normalized["possible_cause"] = _add_multi_fact(
            normalized.get("possible_cause", []),
            "possible_cause",
            "眼科相关",
        )
        normalized["possible_cause"] = _add_multi_fact(
            normalized.get("possible_cause", []),
            "possible_cause",
            "神经系统相关",
        )
        normalized["consult_department"] = _add_multi_fact(
            normalized.get("consult_department", []),
            "consult_department",
            "眼科",
        )
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "存在明显危险信号",
            "danger_signal",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "立即急诊",
            "urgency_level",
        )
        normalized["consult_department"] = _add_multi_fact(
            normalized.get("consult_department", []),
            "consult_department",
            "急诊",
        )
        record_override(
            "vision-loss-risk",
            ("danger_signal", "urgency_level", "possible_cause", "consult_department"),
            "Applied acute-vision-loss safety override",
        )

    if _SKIN_RISK_SIGNS & risk_set:
        normalized["possible_cause"] = _add_multi_fact(
            normalized.get("possible_cause", []),
            "possible_cause",
            "皮肤或过敏相关",
        )
        normalized["consult_department"] = _add_multi_fact(
            normalized.get("consult_department", []),
            "consult_department",
            "皮肤科/变态反应科",
        )
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "可能存在危险信号",
            "danger_signal",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "尽快线下就医",
            "urgency_level",
        )
        record_override(
            "skin-risk",
            ("danger_signal", "urgency_level", "possible_cause", "consult_department"),
            "Applied severe-rash safety override",
        )

    if _STRONG_BLEEDING_SIGNS & risk_set:
        normalized["danger_signal"] = _update_single_select(
            normalized.get("danger_signal", []),
            "存在明显危险信号",
            "danger_signal",
        )
        normalized["urgency_level"] = _update_single_select(
            normalized.get("urgency_level", []),
            "立即急诊",
            "urgency_level",
        )
        normalized["consult_department"] = _add_multi_fact(
            normalized.get("consult_department", []),
            "consult_department",
            "急诊",
        )
        record_override(
            "severe-bleeding",
            ("danger_signal", "urgency_level", "consult_department"),
            "Applied severe-bleeding safety override",
        )

    normalized["possible_cause"] = _cleanup_multi_facts("possible_cause", normalized.get("possible_cause", []))
    normalized["consult_department"] = _cleanup_multi_facts("consult_department", normalized.get("consult_department", []))
    normalized["risk_signal"] = _order_by_schema("risk_signal", normalized.get("risk_signal", []))
    normalized["low_risk_factor"] = _order_by_schema("low_risk_factor", normalized.get("low_risk_factor", []))
    return _dedupe_keep_order(warnings), overrides


def _contains_any_nonconditional_phrase(texts: Sequence[str], aliases: Sequence[str]) -> bool:
    clauses = _build_clause_pool(texts)
    for clause in clauses:
        for alias in aliases:
            if _alias_positive_in_clause(clause, alias, allow_conditional=False):
                return True
    return False


def _text_has_any_nonconditional(text_corpus: Sequence[str], phrases: Sequence[str]) -> bool:
    return _contains_any_nonconditional_phrase(text_corpus, phrases)


def _text_has_any_raw(text_corpus: Sequence[str], phrases: Sequence[str]) -> bool:
    normalized_phrases = [normalize_text_for_match(phrase) for phrase in phrases if normalize_text_for_match(phrase)]
    if not normalized_phrases:
        return False
    for text in text_corpus:
        normalized_text = normalize_text_for_match(text)
        if any(phrase in normalized_text for phrase in normalized_phrases):
            return True
    return False


def _text_has_pediatric_hint(text_corpus: Sequence[str]) -> bool:
    return _text_has_any_raw(text_corpus, ("孩子", "儿童", "宝宝", "婴儿", "幼儿", "小孩"))


def _text_has_pregnancy_hint(text_corpus: Sequence[str]) -> bool:
    return _text_has_any_raw(text_corpus, ("怀孕", "孕期", "孕妇", "妊娠"))


def _infer_departments_from_causes(
    possible_causes: Sequence[str],
    risk_signals: Sequence[str] | None = None,
    text_corpus: Sequence[str] | None = None,
) -> list[str]:
    inferred: list[str] = []
    cause_to_department = {
        "心血管相关": "心内科",
        "呼吸系统相关": "呼吸科",
        "神经系统相关": "神经内科",
        "消化系统相关": "消化科",
        "感染相关": "全科/普通内科",
        "肌肉骨骼相关": "骨科/康复科",
        "泌尿系统相关": "泌尿外科/肾内科",
        "皮肤或过敏相关": "皮肤科/变态反应科",
        "耳鼻喉相关": "耳鼻喉科",
        "眼科相关": "眼科",
        "妇产/生殖相关": "妇产科",
        "内分泌代谢相关": "内分泌科",
        "血液或免疫相关": "血液科/风湿免疫科",
        "药物或中毒相关": "急诊",
        "压力焦虑或睡眠相关": "心理/精神心理科",
    }
    for cause in possible_causes:
        department = cause_to_department.get(cause)
        if department:
            inferred.append(department)
    return _cleanup_multi_facts("consult_department", inferred)


def _should_add_pediatric_department(
    text_corpus: Sequence[str],
    risk_signals: Sequence[str],
) -> bool:
    if not _text_has_pediatric_hint(text_corpus):
        return False
    pediatric_emergency_signals = {
        "高热",
        "抽搐",
        "意识改变",
        "儿童精神差或反应差",
        "严重出血",
        "持续呕吐或腹泻",
        "脱水表现",
    }
    return not bool(pediatric_emergency_signals & set(risk_signals))


def normalize_model_medical_output(
    model_output: Any,
    *,
    user_text: str = "",
    max_candidates_per_object: int | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    user_explanation = ""
    raw_structured_analysis: dict[str, Any] = {}

    if isinstance(model_output, Mapping):
        explanation_value = model_output.get("user_explanation")
        if explanation_value is not None:
            user_explanation = " ".join(flatten_raw_value(explanation_value)).strip()

        structured_value = model_output.get("structured_analysis")
        if isinstance(structured_value, Mapping):
            raw_structured_analysis = dict(structured_value)
        else:
            top_level_fields = {
                schema.object_id: model_output[schema.object_id]
                for schema in MEDICAL_OBJECTS
                if schema.object_id in model_output
            }
            if top_level_fields:
                raw_structured_analysis = top_level_fields
                warnings.append("structured_analysis missing or not a dict; used top-level medical fields")
            else:
                warnings.append("structured_analysis missing or not a dict")
    else:
        user_explanation = " ".join(flatten_raw_value(model_output)).strip()
        warnings.append("model_output is not a dict; structured_analysis unavailable")

    from_model_fields = _empty_patch_map()
    from_user_text = _extract_user_text_patch_facts(user_text)
    field_status = _empty_field_status()
    normalized: dict[str, list[str]] = _empty_patch_map()
    rows: list[dict[str, Any]] = []

    for schema in MEDICAL_OBJECTS:
        raw_value = raw_structured_analysis.get(schema.object_id)
        is_missing = schema.object_id not in raw_structured_analysis
        if is_missing:
            warnings.append(f"Missing field: {schema.object_id}")
        model_facts, used_fallback = _normalize_medical_fact_detailed(
            schema.object_id,
            raw_value,
            user_text="",
        )
        field_status[schema.object_id]["missing"] = is_missing
        field_status[schema.object_id]["used_fallback"] = used_fallback
        from_model_fields[schema.object_id] = list(model_facts)
        normalized[schema.object_id] = list(model_facts)
        rows.append(
            {
                "object_id": schema.object_id,
                "object_label": schema.label,
                "raw_value": raw_value,
                "model_field_facts": list(model_facts),
                "user_text_patch_facts": list(from_user_text.get(schema.object_id, [])),
                "final_normalized_facts": list(model_facts),
                "normalized_facts": list(model_facts),
            }
        )

    _merge_patch_map_into_normalized(normalized, from_user_text)
    text_corpus = _collect_text_corpus(model_output, raw_structured_analysis, user_explanation)
    if user_text.strip():
        text_corpus.append(user_text)
    safety_warnings, safety_overrides = _apply_safety_overrides(normalized, text_corpus=text_corpus)
    warnings.extend(safety_warnings)
    inferred_departments_patch = {"consult_department": []}
    inferred_departments = _infer_departments_from_causes(
        normalized.get("possible_cause", []),
        risk_signals=normalized.get("risk_signal", []),
        text_corpus=text_corpus,
    )
    if _should_add_pediatric_department(text_corpus, normalized.get("risk_signal", [])):
        inferred_departments = _add_multi_fact(inferred_departments, "consult_department", "儿科")
    for department in inferred_departments:
        if department not in normalized.get("consult_department", []):
            normalized["consult_department"] = _add_multi_fact(
                normalized.get("consult_department", []),
                "consult_department",
                department,
            )
            inferred_departments_patch["consult_department"].append(department)

    for schema in MEDICAL_OBJECTS:
        facts = normalized.get(schema.object_id, [])
        if schema.mode == "multi":
            facts = _cleanup_multi_facts(schema.object_id, facts)
        else:
            facts = facts[:1]
        normalized[schema.object_id] = _truncate_candidates(
            schema.object_id,
            facts,
            max_candidates_per_object,
        )
        if schema.mode == "single" and not normalized[schema.object_id]:
            normalized[schema.object_id] = list(schema.fallback)

    for row in rows:
        final_facts = list(normalized[row["object_id"]])
        row["final_normalized_facts"] = final_facts
        row["normalized_facts"] = final_facts

    return {
        "user_explanation": user_explanation,
        "raw_structured_analysis": raw_structured_analysis,
        "normalized": normalized,
        "rows": rows,
        "field_status": field_status,
        "patches": {
            "from_model_fields": from_model_fields,
            "from_user_text": from_user_text,
            "safety_overrides": safety_overrides,
            "inferred_departments": inferred_departments_patch,
        },
        "warnings": _dedupe_keep_order(warnings),
    }


def normalize_all_models_medical_outputs(
    model_outputs: Mapping[str, Any],
    *,
    user_text: str = "",
) -> dict[str, dict[str, Any]]:
    return {
        model_name: normalize_model_medical_output(model_output, user_text=user_text)
        for model_name, model_output in model_outputs.items()
    }


def build_medical_fact_table(
    normalized_all: Mapping[str, Mapping[str, Any]],
    source: str = "normalized",
    exclude_fallbacks: bool = False,
) -> list[dict[str, Any]]:
    if source not in {"normalized", "from_model_fields", "from_user_text"}:
        raise ValueError(f"Invalid medical fact table source: {source}")
    table: list[dict[str, Any]] = []
    for model_name, result in normalized_all.items():
        if source == "normalized":
            fact_map = result.get("normalized", {})
        else:
            fact_map = (result.get("patches", {}) or {}).get(source, {})
        field_status = result.get("field_status", {}) or {}
        for schema in MEDICAL_OBJECTS:
            facts = list(fact_map.get(schema.object_id, []))
            if (
                source == "from_model_fields"
                and exclude_fallbacks
                and bool((field_status.get(schema.object_id, {}) or {}).get("used_fallback"))
            ):
                facts = []
            table.append(
                {
                    "model": model_name,
                    "object_id": schema.object_id,
                    "object_label": schema.label,
                    "facts": facts,
                }
            )
    return table


__all__ = [
    "build_medical_fact_table",
    "flatten_raw_value",
    "get_medical_objects",
    "normalize_all_models_medical_outputs",
    "normalize_medical_fact",
    "normalize_model_medical_output",
    "normalize_text_for_match",
    "split_clauses",
]


if __name__ == "__main__":
    demo_cases = [
        {
            "name": "chest_case",
            "user_text": "我今年22岁，胸闷，爬楼梯时明显，休息后缓解，偶尔心慌，没有发烧咳嗽。",
            "model_output": {
                "user_explanation": "需要结合症状进一步判断。",
                "structured_analysis": {
                    "urgency_level": "可以先观察几天",
                    "possible_cause": ["压力大、熬夜可能有关"],
                },
            },
        },
        {
            "name": "neuro_case",
            "user_text": "突然说话不清，一侧肢体无力。",
            "model_output": {
                "user_explanation": "存在需要重视的神经系统风险。",
                "structured_analysis": {},
            },
        },
        {
            "name": "conditional_case",
            "user_text": "",
            "model_output": {
                "user_explanation": "目前普通门诊即可，如胸痛加重应立即急诊。",
                "structured_analysis": {
                    "urgency_level": "目前普通门诊即可，如胸痛加重应立即急诊"
                },
            },
        },
    ]

    for demo in demo_cases:
        print(f"=== {demo['name']} ===")
        print(
            json.dumps(
                normalize_model_medical_output(
                    demo["model_output"],
                    user_text=demo["user_text"],
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
