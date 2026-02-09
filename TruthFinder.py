from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Iterable, Optional
import math
import re

# 1) Soft normalize: raw -> List[candidate fact]
ZH_ALIAS_SOFT = {
    "站点": "网站",
    "网页": "网站",
    "網站": "网站",
}

_SPLIT_RE = re.compile(r"[，,;；/｜|、\n]+|(?:\s+)|(?:或者)|(?:或是)|(?:\b或\b)")
_PARENS_RE = re.compile(r"[\(\（].*?[\)\）]")
_PREFIX_PATTERNS = [
    r"^指的是[:：]?", r"^指为[:：]?", r"^表示[:：]?", r"^意为[:：]?",
    r"^意思是[:：]?", r"^一种[:：]?", r"^用于[:：]?", r"^用来[:：]?",
    r"^用於[:：]?", r"^即[:：]?", r"^也称[:：]?", r"^又称[:：]?", r"^亦称[:：]?"
]
_PREFIX_RE = re.compile("|".join(_PREFIX_PATTERNS))
_KEEP_CORE_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")


def _clean_piece(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = _PARENS_RE.sub("", s).strip()
    s = _PREFIX_RE.sub("", s).strip()
    s = s.strip(" \"'“”‘’。.!！?？:：")
    if not s:
        return ""
    s = _KEEP_CORE_RE.sub("", s).strip()
    s = ZH_ALIAS_SOFT.get(s, s)
    return s


def normalize_meaning_zh_soft(raw: str, top_n: int = 3) -> List[str]:
    """
    软归一化：返回 1~top_n 个候选（fact candidates），尽量保留差异。
    """
    if not raw or not str(raw).strip():
        return []
    raw2 = _PARENS_RE.sub("", str(raw).strip())
    parts = [p.strip() for p in _SPLIT_RE.split(raw2) if p and p.strip()]
    if not parts:
        parts = [raw2]

    cleaned: List[str] = []
    seen: Set[str] = set()
    for p in parts:
        c = _clean_piece(p)
        if c and c not in seen:
            seen.add(c)
            cleaned.append(c)

    if not cleaned:
        return []

    # 排序：优先“包含汉字”的候选，其次更短更像词条（避免长句）
    cleaned.sort(
        key=lambda x: (1 if re.search(r"[\u4e00-\u9fff]", x) else 0, -len(x)),
        reverse=True
    )
    return cleaned[:top_n]


def display_norm_candidates(cands: List[str]) -> str:
    if not cands:
        return "(空)"
    return "｜".join(cands)

# 2) Similarity + implication (for facts under same object)
def _char_ngrams(s: str, n: int = 2) -> Set[str]:
    s = (s or "").strip()
    if not s:
        return set()
    # 对中文，2-gram 的区分度一般够用
    if len(s) < n:
        return {s}
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def fact_similarity(f1: str, f2: str) -> float:
    if f1 == f2:
        return 1.0
    return jaccard(_char_ngrams(f1, 2), _char_ngrams(f2, 2))


def build_implication_matrix(facts: List[str]) -> Dict[Tuple[str, str], float]:
    """
    返回 imp[(f_i, f_j)] ∈ [0,1]：
    - 越相近，imp 越大（正向支持）
    - 这里不显式建负向 imp；冲突用 (1-sim) 在更新时做惩罚
    """
    imp: Dict[Tuple[str, str], float] = {}
    for i in range(len(facts)):
        for j in range(len(facts)):
            if i == j:
                continue
            sim = fact_similarity(facts[i], facts[j])
            # 相似度太低就当作没有支持关系，避免引入噪声
            if sim >= 0.35:
                imp[(facts[i], facts[j])] = sim
    return imp


# 3) TruthFinder core with rho(dependency) + gamma(dampening)
def _tau(t: float) -> float:
    t = min(max(t, 1e-6), 1 - 1e-6)
    return -math.log(1 - t)

def _sigmoid(x: float) -> float:
    # 避免溢出
    if x >= 60:
        return 1.0
    if x <= -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def compute_rho_dependency(
    models: List[str],
    obj_fact_support: Dict[Tuple[str, str], Set[str]],
) -> Dict[Tuple[str, str], float]:
    """
    估计模型间“相互依赖程度” rho(w1,w2) ∈ [0,1]。
    我的场景：若两个模型经常给出相同/相近的 fact（在同一批 object 上），
    则它们的投票不能当作独立证据。

    简化做法：把每个模型提供的 (object,fact) 当作集合，算 Jaccard。
    """
    model_sets: Dict[str, Set[Tuple[str, str]]] = {m: set() for m in models}
    for (obj, fact), ws in obj_fact_support.items():
        for w in ws:
            model_sets[w].add((obj, fact))

    rho: Dict[Tuple[str, str], float] = {}
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                continue
            a = model_sets[models[i]]
            b = model_sets[models[j]]
            rho[(models[i], models[j])] = jaccard(a, b)
    return rho


@dataclass
class TruthFinderConfig:
    # 初始模型可信度
    t0: float = 0.75
    # 用于把 Σtau 映射到 s 的强度
    beta: float = 0.35
    # 依赖衰减因子 gamma：越大越“保守”，越不相信相关模型的重复投票
    gamma: float = 0.35
    # implication 强度（对同一 object 下相近事实的“互相加分”）
    alpha_imp: float = 0.25
    # 冲突惩罚强度（对同一 object 下互斥/很不相似事实的“互相扣分”）
    alpha_conflict: float = 0.15
    # 每个释义保留候选数量
    topn_candidates: int = 3
    # 迭代停止阈值：1 - cosine(old_t, new_t) < delta
    delta: float = 1e-4
    max_iter: int = 25


def truthfinder_run(
    models: List[str],
    sentence_id: str,
    keywords: List[str],
    results: Dict[str, dict],
    cfg: TruthFinderConfig = TruthFinderConfig()
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], Dict[str, float]], Dict[Tuple[str, str], List[str]]]:
    """
    返回：
    1) t_score[model] = 模型可信度
    2) s_score[(sentence_id,kw)][fact] = 该 object 下每个 fact 的置信度
    3) cand_map[(sentence_id,kw)] = 该 object 的候选 fact 列表（用于展示/调试）
    """

    # ----- 1) 构造 object / fact / providers -----
    # object: (sentence_id, keyword)
    objects: List[Tuple[str, str]] = [(sentence_id, kw) for kw in keywords]

    # obj_facts[o] = list of candidate facts for that object (dedup)
    obj_facts: Dict[Tuple[str, str], List[str]] = {o: [] for o in objects}
    # providers[(obj,fact)] = set(models)
    obj_fact_support: Dict[Tuple[str, str], Set[str]] = {}

    # results[model]["keywords"] = [{"keyword":..., "meaning_zh":...}, ...]
    for m in models:
        payload = results.get(m, {}) or {}
        rows = payload.get("keywords", []) or []
        # 建一个 kw->meaning 的映射，容错
        kw2raw = {}
        for r in rows:
            k = (r.get("keyword") or "").strip()
            v = (r.get("meaning_zh") or "").strip()
            if k:
                kw2raw[k] = v

        for kw in keywords:
            o = (sentence_id, kw)
            raw = kw2raw.get(kw, "")
            cands = normalize_meaning_zh_soft(raw, top_n=cfg.topn_candidates)
            # 一个模型对一个关键词可能给出多个候选（例如 “执行｜进行｜实施”）
            # 我们认为它“支持”这些候选（弱支持）
            for fact in cands:
                obj_fact_support.setdefault((o, fact), set()).add(m)

            # 维护 object 的候选全集（去重）
            for fact in cands:
                if fact not in obj_facts[o]:
                    obj_facts[o].append(fact)

    # 若某些 object 一个候选都没有（模型都空），补一个占位，避免后续空集合
    for o in objects:
        if not obj_facts[o]:
            obj_facts[o] = ["(空)"]
            obj_fact_support.setdefault((o, "(空)"), set()).update(set(models))

    # ----- 2) 计算 rho 依赖矩阵（模型间相似度/依赖） -----
    rho = compute_rho_dependency(models, obj_fact_support)

    # 给每个模型一个“平均依赖度”（与其他模型的平均 rho），用于衰减它的贡献
    dep_avg: Dict[str, float] = {}
    for w in models:
        vals = [rho.get((w, u), 0.0) for u in models if u != w]
        dep_avg[w] = sum(vals) / len(vals) if vals else 0.0

    # ----- 3) 初始化 t(w) -----
    t: Dict[str, float] = {w: cfg.t0 for w in models}

    # F(w)：模型 w 提供了哪些 (o,fact)（用于更新 t）
    Fw: Dict[str, List[Tuple[Tuple[str, str], str]]] = {w: [] for w in models}
    for (o, fact), ws in obj_fact_support.items():
        for w in ws:
            Fw[w].append((o, fact))

    # s_score[o][fact] = confidence
    s_score: Dict[Tuple[str, str], Dict[str, float]] = {o: {} for o in objects}

    # ----- 4) 迭代更新：先算 s，再算 t -----
    for _ in range(cfg.max_iter):
        old_vec = [t[w] for w in models]

        tau = {}
        for w in models:
            base = _tau(t[w])
            # 依赖衰减：越依赖越扣（gamma）
            # dep_avg=0 -> 不扣；dep_avg=1 -> 最多扣 gamma
            tau[w] = base * max(0.05, (1.0 - cfg.gamma * dep_avg[w]))

        # 4.1 更新每个 object 下每个 fact 的置信度
        for o in objects:
            facts = obj_facts[o]
            imp = build_implication_matrix(facts)

            # 先算“证据项”：来自支持该 fact 的模型贡献
            base_sigma: Dict[str, float] = {}
            for f in facts:
                ws = obj_fact_support.get((o, f), set())
                base_sigma[f] = sum(tau[w] for w in ws)

            # 再加上 implication / conflict 的影响（同一 object 内部）
            for f in facts:
                # 正向：相似事实互相加分
                imp_bonus = 0.0
                # 负向：很不相似/冲突的事实互相扣分
                conf_pen = 0.0

                for g in facts:
                    if g == f:
                        continue
                    sim = fact_similarity(f, g)
                    # g 的“强度”用它当前的证据项近似（避免需要上轮 s）
                    g_strength = base_sigma.get(g, 0.0)

                    if sim >= 0.35:
                        imp_bonus += imp.get((g, f), 0.0) * g_strength
                    elif sim <= 0.15:
                        conf_pen += (1.0 - sim) * g_strength

                score = base_sigma[f] + cfg.alpha_imp * imp_bonus - cfg.alpha_conflict * conf_pen
                # 映射到 [0,1]
                s_score[o][f] = _sigmoid(cfg.beta * score)

        # 4.2 更新 t(w)：取其提供的 (o,fact) 的平均 s
        new_t: Dict[str, float] = {}
        for w in models:
            pairs = Fw.get(w, [])
            if not pairs:
                new_t[w] = t[w]
                continue
            vals = []
            for (o, f) in pairs:
                vals.append(s_score.get(o, {}).get(f, 0.0))
            new_t[w] = sum(vals) / len(vals) if vals else t[w]

        t = new_t
        new_vec = [t[w] for w in models]
        if 1.0 - _cosine(old_vec, new_vec) < cfg.delta:
            break

    # 输出候选表（用于 app 展示/调试）
    cand_map = {o: obj_facts[o] for o in objects}
    return t, s_score, cand_map


def pick_truth_per_keyword(
    sentence_id: str,
    keywords: List[str],
    s_score: Dict[Tuple[str, str], Dict[str, float]],
    top_k: int = 2,
    margin: float = 0.03
) -> List[dict]:
    """
    对每个 keyword（object）选出 top 事实。
    若 top1 与 top2 差距很小（<= margin），返回两个，避免“硬选唯一真值”。
    """
    out = []
    for kw in keywords:
        o = (sentence_id, kw)
        facts_conf = list((s_score.get(o, {}) or {}).items())
        if not facts_conf:
            out.append({"keyword": kw, "truth": ["(空)"], "conf": [0.0]})
            continue
        facts_conf.sort(key=lambda x: x[1], reverse=True)

        if len(facts_conf) == 1:
            out.append({"keyword": kw, "truth": [facts_conf[0][0]], "conf": [facts_conf[0][1]]})
            continue

        (f1, c1), (f2, c2) = facts_conf[0], facts_conf[1]
        if c1 - c2 <= margin:
            picked = facts_conf[:max(2, top_k)]
        else:
            picked = facts_conf[:1]

        out.append({
            "keyword": kw,
            "truth": [f for f, _ in picked],
            "conf": [float(c) for _, c in picked]
        })
    return out


def rank_models_by_trust(t_score: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(t_score.items(), key=lambda x: x[1], reverse=True)
