from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import math
from normalize import normalize_meaning_zh_soft

# 1) Similarity / implication utilities

def _char_ngrams(s: str, n: int = 2) -> Set[str]:
    s = (s or "").strip()
    if not s:
        return set()
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

def build_implication_matrix(
    facts: List[str],
    sim_threshold: float = 0.45
) -> Dict[Tuple[str, str], float]:
    """
    imp[(g,f)] ∈ [0,1]：g 对 f 的正向支持强度（由相似度近似）。
    - threshold 提高（默认 0.45）：减少“不同含义但字面相似”造成的误传播
    """
    imp: Dict[Tuple[str, str], float] = {}
    for i in range(len(facts)):
        for j in range(len(facts)):
            if i == j:
                continue
            g, f = facts[i], facts[j]
            sim = fact_similarity(g, f)
            if sim >= sim_threshold:
                imp[(g, f)] = sim
    return imp

def build_conflict_matrix(
    facts: List[str],
    sim_threshold: float = 0.15
) -> Dict[Tuple[str, str], float]:
    """
    conf[(g,f)] ∈ [0,1]：g 对 f 的冲突强度（由低相似度近似）。
    - 仅在 sim <= threshold 时记录 1-sim
    """
    conf: Dict[Tuple[str, str], float] = {}
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            fi, fj = facts[i], facts[j]
            sim = fact_similarity(fi, fj)
            if sim <= sim_threshold:
                strength = 1.0 - sim
                conf[(fi, fj)] = strength
                conf[(fj, fi)] = strength
    return conf

# 2) Core math utilities
def _tau(t: float) -> float:
    t = min(max(t, 1e-6), 1 - 1e-6)
    return -math.log(1 - t)

def _sigmoid(x: float) -> float:
    # avoid overflow
    if x >= 60:
        return 1.0
    if x <= -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# 3) rho dependency (model correlation)

def compute_rho_dependency_top1(
    models: List[str],
    top1_choice: Dict[str, Dict[Tuple[str, str], str]],
) -> Dict[Tuple[str, str], float]:
    """
    依赖度 rho(w1,w2) ∈ [0,1]
    这里用“每个 object 的 top1 fact”来计算 Jaccard：
    - 避免 soft-normalize 多候选导致 rho 虚高
    top1_choice[model][obj] = top1_fact
    """
    model_sets: Dict[str, Set[Tuple[Tuple[str, str], str]]] = {m: set() for m in models}

    for m in models:
        for obj, f in (top1_choice.get(m, {}) or {}).items():
            model_sets[m].add((obj, f))

    rho: Dict[Tuple[str, str], float] = {}
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                continue
            w1, w2 = models[i], models[j]
            rho[(w1, w2)] = jaccard(model_sets[w1], model_sets[w2])
    return rho


# 4) Config
@dataclass
class TruthFinderConfig:
    # 初始模型可信度
    t0: float = 0.75

    # Σ(evidence) -> s 的映射强度
    beta: float = 0.35

    # 依赖衰减强度（越大越不相信“相关模型的重复投票”）
    gamma: float = 0.30

    # implication 强度（相似事实互相支持）
    alpha_imp: float = 0.20

    # conflict 强度（强冲突惩罚）
    alpha_conflict: float = 0.10

    # 每个释义保留候选数量
    topn_candidates: int = 3

    # implication 相似阈值（越高越保守）
    imp_sim_threshold: float = 0.45

    # conflict 相似阈值（越低越保守）
    conflict_sim_threshold: float = 0.15

    # 兼容字段：固定轮数迭代时不再用于提前停止
    delta: float = 1e-4
    max_iter: int = 25

    # 多候选加权的衰减系数（越小越偏向 top1）
    # weight_i ∝ cand_decay^i, i=0..n-1
    cand_decay: float = 0.30

    # tau 最小衰减比例（避免被依赖惩罚到接近 0）
    min_tau_scale: float = 0.20


# 5) Helpers: candidate weighting

def _candidate_weights(n: int, decay: float) -> List[float]:
    """
    生成 n 个候选的权重，保证 sum=1。
    - n=1 -> [1]
    - n>1 -> w_i ∝ decay^i（i=0最强），再归一化
    """
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    decay = min(max(decay, 1e-6), 1.0)
    ws = [decay ** i for i in range(n)]
    s = sum(ws)
    return [w / s for w in ws] if s > 0 else [1.0 / n] * n


# 6) TruthFinder main

def truthfinder_run(
    models: List[str],
    sentence_id: str,
    keywords: List[str],
    results: Dict[str, dict],
    cfg: TruthFinderConfig = TruthFinderConfig()
) -> Tuple[
    Dict[str, float],                               # t_score[model]
    Dict[Tuple[str, str], Dict[str, float]],        # s_score[obj][fact]
    Dict[Tuple[str, str], List[str]]                # cand_map[obj] -> facts
]:
    """
    返回：
    1) t_score[model] = 模型可信度
    2) s_score[(sentence_id,kw)][fact] = 该 object 下每个 fact 的置信度
    3) cand_map[(sentence_id,kw)] = 该 object 的候选 fact 列表（用于展示/调试）
    """

    # ----- 1) objects -----
    objects: List[Tuple[str, str]] = [(sentence_id, kw) for kw in keywords]

    # 候选表
    obj_facts: Dict[Tuple[str, str], List[str]] = {o: [] for o in objects}

    # “加权支持”：support[(obj,fact)][model] = weight
    support: Dict[Tuple[Tuple[str, str], str], Dict[str, float]] = {}

    # 为 rho 计算准备：top1_choice[model][obj] = top1_fact
    top1_choice: Dict[str, Dict[Tuple[str, str], str]] = {m: {} for m in models}

    # ----- 2) build candidates + weighted support -----
    for m in models:
        payload = results.get(m, {}) or {}
        rows = payload.get("keywords", []) or []

        kw2raw: Dict[str, str] = {}
        for r in rows:
            k = (r.get("keyword") or "").strip()
            v = (r.get("meaning_zh") or "").strip()
            if k:
                kw2raw[k] = v

        for kw in keywords:
            o = (sentence_id, kw)
            raw = kw2raw.get(kw, "")

            # 归一化仍由 normalize.py 外部函数提供，这里仅调用结果
            cands = normalize_meaning_zh_soft(raw, top_n=cfg.topn_candidates)
            cands = [c.strip() for c in (cands or []) if c and str(c).strip()]

            # 如果完全空，先跳过；后面统一补 "(空)"
            if not cands:
                continue

            # 记录 top1（用于 rho）
            top1_choice[m][o] = cands[0]

            # 对多个候选分配权重（总和=1）
            ws = _candidate_weights(len(cands), decay=cfg.cand_decay)

            for fact, wgt in zip(cands, ws):
                key = (o, fact)
                support.setdefault(key, {})
                support[key][m] = support[key].get(m, 0.0) + float(wgt)

            # 维护 obj 的候选全集（去重）
            for fact in cands:
                if fact not in obj_facts[o]:
                    obj_facts[o].append(fact)

    # 若某些 object 一个候选都没有（模型都空），补 "(空)"，并让所有模型弱支持它（平均分）
    for o in objects:
        if not obj_facts[o]:
            obj_facts[o] = ["(空)"]
            key = (o, "(空)")
            support.setdefault(key, {})
            for m in models:
                support[key][m] = 1.0 / max(1, len(models))
                top1_choice[m][o] = "(空)"

    # ----- 3) rho dependency -----
    rho = compute_rho_dependency_top1(models, top1_choice)

    # 仍保留“平均依赖度”的做法，但 rho 已不再因多候选虚高
    dep_avg: Dict[str, float] = {}
    for w in models:
        vals = [rho.get((w, u), 0.0) for u in models if u != w]
        dep_avg[w] = sum(vals) / len(vals) if vals else 0.0

    # ----- 4) init t -----
    t: Dict[str, float] = {w: cfg.t0 for w in models}

    # F(w)：用于更新 t 的“加权样本”
    # entries[w] = list of (obj, fact, weight)
    entries: Dict[str, List[Tuple[Tuple[str, str], str, float]]] = {w: [] for w in models}
    for (o, fact), m2w in support.items():
        for w, wgt in m2w.items():
            if wgt > 0:
                entries[w].append((o, fact, float(wgt)))

    # s_score[obj][fact]
    s_score: Dict[Tuple[str, str], Dict[str, float]] = {o: {} for o in objects}
    last_s: Dict[Tuple[str, str], Dict[str, float]] = {o: {} for o in objects}

    # 预构造显式矩阵：implication / conflict 都在迭代前固定
    imp_mats: Dict[Tuple[str, str], Dict[Tuple[str, str], float]] = {}
    conf_mats: Dict[Tuple[str, str], Dict[Tuple[str, str], float]] = {}
    for o in objects:
        facts = obj_facts[o]
        imp_mats[o] = build_implication_matrix(facts, sim_threshold=cfg.imp_sim_threshold)
        conf_mats[o] = build_conflict_matrix(facts, sim_threshold=cfg.conflict_sim_threshold)

    # ----- 5) iterate -----
    # 固定轮数迭代（默认 25 轮），不再提前停止
    for _ in range(cfg.max_iter):

        # 5.1 tau with dependency dampening (global, but rho computed on top1)
        tau: Dict[str, float] = {}
        for w in models:
            base = _tau(t[w])
            scale = 1.0 - cfg.gamma * dep_avg.get(w, 0.0)
            scale = max(cfg.min_tau_scale, scale)
            tau[w] = base * scale

        # 5.2 update s for each object
        for o in objects:
            facts = obj_facts[o]
            imp = imp_mats.get(o, {})
            conf = conf_mats.get(o, {})

            # evidence from weighted votes
            base_sigma: Dict[str, float] = {}
            for f in facts:
                m2w = support.get((o, f), {}) or {}
                base_sigma[f] = sum(tau[m] * wgt for m, wgt in m2w.items())

            # implication/conflict use last_s (avoid same-iteration self-amplification)
            for f in facts:
                imp_bonus = 0.0
                conf_pen = 0.0

                for g in facts:
                    if g == f:
                        continue

                    g_strength = float(last_s.get(o, {}).get(g, 0.0))
                    # implication / conflict 都直接从显式矩阵读取
                    imp_bonus += imp.get((g, f), 0.0) * g_strength
                    conf_pen += conf.get((g, f), 0.0) * g_strength

                score = base_sigma[f] + cfg.alpha_imp * imp_bonus - cfg.alpha_conflict * conf_pen
                s_score[o][f] = _sigmoid(cfg.beta * score)

        # 把本轮 s 复制到 last_s（用于下一轮的 implication）
        for o in objects:
            last_s[o] = dict(s_score.get(o, {}))

        # 5.3 update t(w): weighted average of s(o,f) over what w supported
        new_t: Dict[str, float] = {}
        for w in models:
            es = entries.get(w, [])
            if not es:
                new_t[w] = t[w]
                continue
            num = 0.0
            den = 0.0
            for (o, f, wgt) in es:
                num += float(wgt) * float(s_score.get(o, {}).get(f, 0.0))
                den += float(wgt)
            new_t[w] = (num / den) if den > 0 else t[w]

        t = new_t

    cand_map = {o: obj_facts[o] for o in objects}
    return t, s_score, cand_map


# 7) Picking truth per keyword (same interface)

def pick_truth_per_keyword(
    sentence_id: str,
    keywords: List[str],
    s_score: Dict[Tuple[str, str], Dict[str, float]],
    top_k: int = 2,
    margin: float = 0.03
) -> List[dict]:
    """
    对每个 keyword(object) 选出 top 事实。
    若 top1 与 top2 差距很小（<= margin），返回多个，避免“硬选唯一真值”。
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
            out.append({"keyword": kw, "truth": [facts_conf[0][0]], "conf": [float(facts_conf[0][1])]})
            continue

        (f1, c1), (f2, c2) = facts_conf[0], facts_conf[1]
        if float(c1) - float(c2) <= margin:
            picked = facts_conf[:max(2, top_k)]
        else:
            picked = facts_conf[:1]

        out.append({
            "keyword": kw,
            "truth": [f for f, _ in picked],
            "conf": [float(c) for _, c in picked],
        })
    return out


def rank_models_by_trust(t_score: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(t_score.items(), key=lambda x: x[1], reverse=True)
