from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import string
from typing import Any, Dict, List, Optional, Set, Tuple

from normalize import normalize_meaning_zh_soft


EMPTY_FACT = "(\u7a7a)"


# 1) Similarity / legacy utilities

def _char_ngrams(s: str, n: int = 2) -> Set[str]:
    s = (s or "").strip()
    if not s:
        return set()
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


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


# 2) Translation-aware relation utilities
# Why not "low similarity => conflict"?
# In translation, equivalent glosses can have low lexical overlap
# (e.g. "\u8865\u4e01" vs "\u4fee\u8865\u7a0b\u5e8f"). Low similarity should usually be
# treated as "unknown relation", not contradiction.

_BRACKET_RE = re.compile(r"\([^)]*\)|\uff08[^\uff09]*\uff09|\[[^\]]*\]|\u3010[^\u3011]*\u3011|\{[^}]*\}")
_SPACE_RE = re.compile(r"\s+")
_QUOTE_RE = re.compile(r"[\"'\u201c\u201d\u2018\u2019`]")
_REL_PUNCT_TABLE = str.maketrans(
    "",
    "",
    string.punctuation
    + "\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\uff08\uff09\u3010\u3011\u300a\u300b\u3008\u3009\u300c\u300d\u300e\u300f\u00b7\u2026\u2014\uff5e",
)


# Some phrases are operational wrappers and should be stripped before
# polarity core matching.
_NEED_PREFIXES = ("\u9700\u8981", "\u9700", "\u8981")


def normalize_fact_for_relation(f: str) -> str:
    """
    Light normalization for relation judgement.
    Remove bracketed notes, spaces, punctuation and quotes.
    """
    s = str(f or "").strip()
    if not s:
        return ""
    s = _BRACKET_RE.sub("", s)
    s = _QUOTE_RE.sub("", s)
    s = _SPACE_RE.sub("", s)
    s = s.translate(_REL_PUNCT_TABLE)
    return s.strip().lower()


def same_synonym_group(a: str, b: str, synonym_groups: List[Set[str]]) -> bool:
    na = normalize_fact_for_relation(a)
    nb = normalize_fact_for_relation(b)
    if not na or not nb:
        return False
    for group in synonym_groups:
        if na in group and nb in group:
            return True
    return False


def is_conflict_pair(a: str, b: str, conflict_pairs: List[Tuple[str, str]]) -> bool:
    na = normalize_fact_for_relation(a)
    nb = normalize_fact_for_relation(b)
    if not na or not nb:
        return False
    for x, y in conflict_pairs:
        nx = normalize_fact_for_relation(x)
        ny = normalize_fact_for_relation(y)
        if (na == nx and nb == ny) or (na == ny and nb == nx):
            return True
    return False


def strip_negation_marker(s: str, cfg: "TruthFinderConfig") -> Tuple[str, bool]:
    """
    Return (core_text, has_negative_polarity).

    Translation-oriented rationale:
    many Chinese security terms differ by polarity markers only
    (e.g., "\u8ba4\u8bc1" vs "\u672a\u8ba4\u8bc1"). Detecting this avoids false support
    from containment/literal overlap.
    """
    ns = normalize_fact_for_relation(s)
    if not ns:
        return "", False

    neg_prefixes = tuple(
        sorted(
            {
                normalize_fact_for_relation(x)
                for x in (cfg.negation_prefixes + cfg.negative_markers)
                if normalize_fact_for_relation(x)
            },
            key=len,
            reverse=True,
        )
    )

    has_neg = False
    core = ns

    for p in neg_prefixes:
        if core.startswith(p):
            core = core[len(p) :]
            has_neg = True
            break

    # Remove weak modality wrappers for stable core comparison.
    for pref in _NEED_PREFIXES:
        np = normalize_fact_for_relation(pref)
        if np and core.startswith(np):
            core = core[len(np) :]
            break

    core = core.strip()
    return core, has_neg


def polarity_conflict(a: str, b: str, cfg: "TruthFinderConfig") -> bool:
    """
    Detect polarity mismatch for the same (or nested) core semantics.
    """
    na = normalize_fact_for_relation(a)
    nb = normalize_fact_for_relation(b)
    if not na or not nb:
        return False

    core_a, neg_a = strip_negation_marker(na, cfg)
    core_b, neg_b = strip_negation_marker(nb, cfg)

    if not core_a or not core_b:
        return False

    if core_a == core_b and (neg_a != neg_b):
        return True

    if neg_a != neg_b and (core_a in core_b or core_b in core_a):
        return True

    return False


def containment_relation(a: str, b: str) -> float:
    """
    Weak directional relation for containing terms, not full equivalence.
    Example: "\u8865\u4e01" vs "\u5b89\u5168\u8865\u4e01".
    """
    na = normalize_fact_for_relation(a)
    nb = normalize_fact_for_relation(b)
    if not na or not nb or na == nb:
        return 0.0

    if na in nb or nb in na:
        short_len = min(len(na), len(nb))
        if short_len <= 1:
            return 0.0
        if short_len <= 2:
            return 0.35
        return 0.40
    return 0.0


def _synonym_group_sets(cfg: "TruthFinderConfig") -> List[Set[str]]:
    return [
        {normalize_fact_for_relation(x) for x in grp if normalize_fact_for_relation(x)}
        for grp in cfg.synonym_groups
    ]


def _conflict_pair_list(cfg: "TruthFinderConfig") -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for a, b in cfg.conflict_pairs:
        na = normalize_fact_for_relation(a)
        nb = normalize_fact_for_relation(b)
        if na and nb:
            out.append((na, nb))
    return out


def fact_relation_score(a: str, b: str, cfg: "TruthFinderConfig") -> float:
    """
    Signed relation in [-1, 1].
    Priority:
    1) exact
    2) polarity conflict
    3) explicit conflict
    4) synonym group
    5) containment
    6) literal similarity fallback
    7) no relation
    """
    na = normalize_fact_for_relation(a)
    nb = normalize_fact_for_relation(b)
    if not na or not nb:
        return 0.0

    if na == nb:
        return float(cfg.rel_exact_score)

    if polarity_conflict(na, nb, cfg):
        return -float(cfg.rel_conflict_score)

    conflict_pairs = _conflict_pair_list(cfg)
    if is_conflict_pair(na, nb, conflict_pairs):
        return -float(cfg.rel_conflict_score)

    synonym_groups = _synonym_group_sets(cfg)
    if same_synonym_group(na, nb, synonym_groups):
        return float(cfg.rel_synonym_score)

    contain = containment_relation(na, nb)
    if contain > 0.0:
        return float(cfg.rel_containment_score)

    sim = fact_similarity(na, nb)
    if sim >= cfg.imp_sim_threshold:
        return float(sim * cfg.rel_literal_scale)

    return 0.0


def build_relation_matrix(
    facts: List[str],
    cfg: "TruthFinderConfig",
) -> Dict[Tuple[str, str], float]:
    rel: Dict[Tuple[str, str], float] = {}
    for i in range(len(facts)):
        for j in range(len(facts)):
            if i == j:
                continue
            g, f = facts[i], facts[j]
            score = fact_relation_score(g, f, cfg)
            if score != 0.0:
                rel[(g, f)] = max(-1.0, min(1.0, score))
    return rel


def cluster_relation_score(
    members_g: List[str],
    members_f: List[str],
    cfg: "TruthFinderConfig",
) -> float:
    """
    Compute signed relation between two candidate clusters.

    Translation-oriented rationale:
    cluster representative may hide signal from other members.
    This function uses member-level pair scores and keeps strongest
    contradiction when present.
    """
    pair_scores: List[float] = []
    for g in members_g:
        for f in members_f:
            if normalize_fact_for_relation(g) == normalize_fact_for_relation(f):
                continue
            sc = fact_relation_score(g, f, cfg)
            if sc != 0.0:
                pair_scores.append(sc)

    if not pair_scores:
        return 0.0

    negs = [x for x in pair_scores if x < 0.0]
    if negs:
        return min(negs)

    poss = [x for x in pair_scores if x > 0.0]
    if poss:
        return max(poss)

    return 0.0


def build_cluster_relation_matrix(
    cluster_facts: List[str],
    cluster_members: Dict[str, List[str]],
    cfg: "TruthFinderConfig",
) -> Dict[Tuple[str, str], float]:
    rel: Dict[Tuple[str, str], float] = {}
    for i in range(len(cluster_facts)):
        for j in range(len(cluster_facts)):
            if i == j:
                continue
            g = cluster_facts[i]
            f = cluster_facts[j]
            g_members = cluster_members.get(g, [g])
            f_members = cluster_members.get(f, [f])
            sc = cluster_relation_score(g_members, f_members, cfg)
            if sc != 0.0:
                rel[(g, f)] = max(-1.0, min(1.0, sc))
    return rel


def build_implication_matrix(
    facts: List[str],
    sim_threshold: float = 0.45,
) -> Dict[Tuple[str, str], float]:
    """
    Legacy helper kept for compatibility.
    Prefer build_relation_matrix/build_cluster_relation_matrix.
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
    sim_threshold: float = 0.15,
) -> Dict[Tuple[str, str], float]:
    """
    Legacy helper kept for compatibility.
    NOTE: low lexical similarity is not reliable contradiction in translation.
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


# 3) Core math utilities

def _tau(t: float) -> float:
    t = min(max(t, 1e-6), 1 - 1e-6)
    return -math.log(1 - t)


def _sigmoid(x: float) -> float:
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


# 4) rho dependency and family priors

def compute_rho_dependency_top1(
    models: List[str],
    top1_choice: Dict[str, Dict[Tuple[str, str], str]],
) -> Dict[Tuple[str, str], float]:
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


def infer_model_family(model_name: str) -> str:
    name = (model_name or "").lower()
    if "qwen" in name:
        return "qwen"
    if "gemma" in name:
        return "gemma"
    if "mistral" in name:
        return "mistral"
    if "llama" in name:
        return "llama"
    if "phi" in name:
        return "phi"
    if "deepseek" in name:
        return "deepseek"
    return "unknown"


def _family_for_model(model: str, cfg: "TruthFinderConfig") -> str:
    if model in cfg.model_family and cfg.model_family[model]:
        return str(cfg.model_family[model]).lower()
    return infer_model_family(model)


def compute_dependency_with_family(
    models: List[str],
    observed_rho: Dict[Tuple[str, str], float],
    cfg: "TruthFinderConfig",
) -> Dict[Tuple[str, str], float]:
    dep: Dict[Tuple[str, str], float] = {}
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                continue
            m1, m2 = models[i], models[j]
            observed = float(observed_rho.get((m1, m2), 0.0))

            if not cfg.use_family_dependency:
                dep[(m1, m2)] = observed
                continue

            f1 = _family_for_model(m1, cfg)
            f2 = _family_for_model(m2, cfg)
            if f1 == f2 and f1 != "unknown":
                family_prior = float(cfg.family_dep_same)
            elif f1 == "unknown" or f2 == "unknown":
                family_prior = float(cfg.family_dep_unknown)
            else:
                family_prior = float(cfg.family_dep_different)

            dep[(m1, m2)] = max(observed, family_prior)
    return dep


# 5) Config

DEFAULT_SYNONYM_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("\u6f0f\u6d1e", "\u5b89\u5168\u6f0f\u6d1e", "\u5b89\u5168\u7f3a\u9677", "\u7f3a\u9677"),
    ("\u8865\u4e01", "\u4fee\u8865\u7a0b\u5e8f", "\u4fee\u590d\u8865\u4e01", "\u5b89\u5168\u8865\u4e01"),
    ("\u8ba4\u8bc1", "\u8eab\u4efd\u9a8c\u8bc1", "\u9274\u6743"),
    ("\u6388\u6743", "\u6743\u9650\u6388\u4e88"),
    ("\u5229\u7528", "\u6f0f\u6d1e\u5229\u7528", "\u653b\u51fb\u5229\u7528"),
    ("\u4e25\u91cd\u7a0b\u5ea6", "\u4e25\u91cd\u6027", "\u5371\u5bb3\u7b49\u7ea7"),
    ("\u4fee\u590d", "\u4fee\u8865", "\u4fee\u6b63"),
    ("\u653b\u51fb\u8005", "\u5a01\u80c1\u8005", "\u6076\u610f\u7528\u6237"),
    ("\u6267\u884c", "\u8fd0\u884c", "\u5b9e\u884c"),
    ("\u8fdc\u7a0b", "\u8fdc\u7a0b\u5730"),
)

DEFAULT_CONFLICT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("\u5df2\u4fee\u590d", "\u672a\u4fee\u590d"),
    ("\u5b58\u5728\u5229\u7528", "\u65e0\u5229\u7528\u8bc1\u636e"),
    ("\u5df2\u88ab\u5229\u7528", "\u672a\u88ab\u5229\u7528"),
    ("\u8865\u4e01", "\u8865\u7247"),
    ("\u6f0f\u6d1e\u5229\u7528", "\u5f00\u53d1"),
    ("\u8ba4\u8bc1", "\u672a\u8ba4\u8bc1"),
    ("\u6388\u6743", "\u672a\u6388\u6743"),
    ("\u5141\u8bb8", "\u963b\u6b62"),
    ("\u542f\u7528", "\u7981\u7528"),
    ("\u6210\u529f", "\u5931\u8d25"),
)


@dataclass
class TruthFinderConfig:
    # Initial trust score per model.
    t0: float = 0.75

    # Legacy name from existing callers. It is score->sigmoid scaling strength.
    # Equivalent to score_scale in this implementation.
    beta: float = 0.35

    # Legacy dependency dampening field kept for compatibility.
    # Prefer dep_dampening when provided.
    gamma: float = 0.30
    dep_dampening: Optional[float] = None

    # Positive and negative relation strengths in iterative score update.
    alpha_imp: float = 0.20
    alpha_conflict: float = 0.10

    # Candidate extraction config.
    topn_candidates: int = 3
    imp_sim_threshold: float = 0.45
    conflict_sim_threshold: float = 0.15

    # Iteration and stopping.
    delta: float = 1e-4
    abs_delta: float = 1e-4
    max_iter: int = 25
    early_stop: bool = True
    # Initialize last_s to neutral confidence so relation can work from round-1.
    init_last_s: float = 0.5
    # Require at least a few iterations before early-stop.
    min_iter: int = 2

    # soft-candidates weighting (top1 gets larger weight).
    cand_decay: float = 0.30
    # "top1": only top1 candidate contributes source support (default, closer to classic TruthFinder).
    # "soft": all candidates contribute weighted support.
    support_mode: str = "top1"

    # Lower bound to avoid dependency dampening driving tau close to zero.
    min_tau_scale: float = 0.40

    # Signed relation scores.
    rel_exact_score: float = 0.90
    rel_conflict_score: float = 0.70
    rel_synonym_score: float = 0.70
    rel_containment_score: float = 0.40
    rel_literal_scale: float = 0.60

    # Negation markers for polarity conflict detection.
    negation_prefixes: Tuple[str, ...] = (
        "\u4e0d",
        "\u672a",
        "\u65e0",
        "\u975e",
        "\u6ca1",
        "\u6ca1\u6709",
        "\u65e0\u9700",
        "\u4e0d\u80fd",
        "\u65e0\u6cd5",
        "\u7981\u7528",
    )
    negative_markers: Tuple[str, ...] = (
        "\u672a",
        "\u65e0",
        "\u4e0d",
        "\u975e",
        "\u6ca1\u6709",
        "\u65e0\u9700",
        "\u4e0d\u80fd",
        "\u65e0\u6cd5",
    )

    # Clustering controls.
    merge_literal_threshold: float = 0.82
    merge_containment: bool = False

    # Extensible rule knowledge.
    synonym_groups: Tuple[Tuple[str, ...], ...] = DEFAULT_SYNONYM_GROUPS
    conflict_pairs: Tuple[Tuple[str, str], ...] = DEFAULT_CONFLICT_PAIRS

    # Model family dependency priors.
    # Why needed: same-family models are not independent evidence sources.
    model_family: Dict[str, str] = field(default_factory=dict)
    family_dep_same: float = 0.50
    family_dep_unknown: float = 0.10
    family_dep_different: float = 0.0
    use_family_dependency: bool = True

    # Trust prior smoothing.
    # Why needed: one sentence often has few keywords, raw updates are noisy.
    use_trust_prior: bool = True
    trust_prior_default: float = 0.75
    trust_prior_by_model: Dict[str, float] = field(default_factory=dict)
    trust_prior_strength: float = 2.0

    # Debug options.
    debug_relations: bool = False
    jsonable_debug: bool = True

    # Placeholder used when all models provide empty candidate for an object.
    empty_fact: str = EMPTY_FACT


# 6) Helpers

def _candidate_weights(n: int, decay: float) -> List[float]:
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    decay = min(max(decay, 1e-6), 1.0)
    ws = [decay**i for i in range(n)]
    s = sum(ws)
    return [w / s for w in ws] if s > 0 else [1.0 / n] * n


def _get_dep_dampening(cfg: TruthFinderConfig) -> float:
    return float(cfg.gamma if cfg.dep_dampening is None else cfg.dep_dampening)


def get_trust_prior(model: str, cfg: TruthFinderConfig) -> float:
    if model in cfg.trust_prior_by_model:
        p = float(cfg.trust_prior_by_model[model])
    else:
        p = float(cfg.trust_prior_default)
    return min(max(p, 0.01), 0.99)


def choose_cluster_representative(
    members: List[str],
    support_hint: Optional[Dict[str, float]],
    cfg: TruthFinderConfig,
) -> str:
    """
    Choose a stable and display-friendly representative for a candidate cluster.

    Translation-oriented rationale:
    representative should prefer canonical terms (synonym dictionary order),
    then stronger model support, then shorter/general form for readability.
    """
    if not members:
        return cfg.empty_fact

    norm_to_member: Dict[str, str] = {}
    member_order: Dict[str, int] = {}
    for idx, m in enumerate(members):
        nm = normalize_fact_for_relation(m)
        if nm and nm not in norm_to_member:
            norm_to_member[nm] = m
        member_order[m] = idx

    # 1) Canonical synonym order.
    for group in cfg.synonym_groups:
        for term in group:
            nt = normalize_fact_for_relation(term)
            if nt in norm_to_member:
                return norm_to_member[nt]

    # 2) Strongest support hint.
    if support_hint:
        best = max(
            members,
            key=lambda x: (
                float(support_hint.get(x, 0.0)),
                -member_order[x],
            ),
        )
        if float(support_hint.get(best, 0.0)) > 0.0:
            return best

    # 3) Shorter/general form.
    best = min(
        members,
        key=lambda x: (
            len(normalize_fact_for_relation(x)),
            member_order[x],
        ),
    )
    return best


def _should_merge_facts(a: str, b: str, cfg: TruthFinderConfig) -> bool:
    na = normalize_fact_for_relation(a)
    nb = normalize_fact_for_relation(b)
    if not na or not nb:
        return False

    if na == nb:
        return True

    if polarity_conflict(na, nb, cfg):
        return False

    groups = _synonym_group_sets(cfg)
    if same_synonym_group(na, nb, groups):
        return True

    conflicts = _conflict_pair_list(cfg)
    if is_conflict_pair(na, nb, conflicts):
        return False

    if fact_similarity(na, nb) >= cfg.merge_literal_threshold:
        return True

    if cfg.merge_containment and containment_relation(na, nb) > 0.0:
        return True

    return False


def cluster_facts_for_object(
    facts: List[str],
    cfg: TruthFinderConfig,
    support_hint: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    """
    Cluster candidates with Union-Find (disjoint set).

    Compared with greedy clustering, union-find preserves transitive merge:
    if A~B and B~C then A/B/C end up in one cluster even if A and C are not
    directly similar enough.
    """
    uniq: List[str] = []
    seen: Set[str] = set()
    for f in facts:
        k = str(f or "").strip()
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(k)

    if not uniq:
        return [], {}, {}

    n = len(uniq)
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    for i in range(n):
        for j in range(i + 1, n):
            if _should_merge_facts(uniq[i], uniq[j], cfg):
                union(i, j)

    root_to_members: Dict[int, List[str]] = {}
    for idx, fact in enumerate(uniq):
        r = find(idx)
        root_to_members.setdefault(r, []).append(fact)

    # Keep deterministic cluster order by first appearance in uniq.
    root_order = sorted(root_to_members.keys(), key=lambda r: uniq.index(root_to_members[r][0]))

    reps: List[str] = []
    fact_to_cluster: Dict[str, str] = {}
    cluster_members: Dict[str, List[str]] = {}

    for r in root_order:
        members = root_to_members[r]
        rep = choose_cluster_representative(members, support_hint=support_hint, cfg=cfg)

        # Safety: ensure representative exists in member list.
        if rep not in members:
            rep = members[0]

        reps.append(rep)
        cluster_members[rep] = list(members)
        for m in members:
            fact_to_cluster[m] = rep

    return reps, fact_to_cluster, cluster_members


def make_jsonable_debug(debug_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert tuple-key Python debug structures into JSON-serializable shapes.
    """
    out: Dict[str, Any] = {
        "iter_count": int(debug_info.get("iter_count", 0)),
        "t_history": list(debug_info.get("t_history", [])),
        "change_history": list(debug_info.get("change_history", [])),
        "abs_change_history": list(debug_info.get("abs_change_history", [])),
        "dep_avg": dict(debug_info.get("dep_avg", {})),
        "support_mode": debug_info.get("support_mode"),
        "min_iter": debug_info.get("min_iter"),
        "init_last_s": debug_info.get("init_last_s"),
    }

    rho_dict = debug_info.get("rho", {}) or {}
    out["rho"] = [
        {"model_a": a, "model_b": b, "value": float(v)}
        for (a, b), v in sorted(rho_dict.items(), key=lambda x: (x[0][0], x[0][1]))
    ]

    dep_dict = debug_info.get("dependency", {}) or {}
    out["dependency"] = [
        {"model_a": a, "model_b": b, "value": float(v)}
        for (a, b), v in sorted(dep_dict.items(), key=lambda x: (x[0][0], x[0][1]))
    ]

    cluster_members = debug_info.get("cluster_members", {}) or {}
    cluster_rows: List[dict] = []
    for (sid, kw), rep_map in sorted(cluster_members.items(), key=lambda x: (x[0][0], x[0][1])):
        clusters = [
            {"representative": rep, "members": list(members)}
            for rep, members in sorted(rep_map.items(), key=lambda x: x[0])
        ]
        cluster_rows.append(
            {
                "sentence_id": sid,
                "keyword": kw,
                "clusters": clusters,
            }
        )
    out["cluster_members"] = cluster_rows

    support = debug_info.get("support", {}) or {}
    support_rows: List[dict] = []
    for ((sid, kw), fact), by_model in sorted(
        support.items(), key=lambda x: (x[0][0][0], x[0][0][1], x[0][1])
    ):
        by_model_clean = {m: float(w) for m, w in by_model.items()}
        support_rows.append(
            {
                "sentence_id": sid,
                "keyword": kw,
                "fact": fact,
                "support_by_model": by_model_clean,
                "support_weight": float(sum(by_model_clean.values())),
            }
        )
    out["support"] = support_rows

    if "relation_mats" in debug_info:
        rel_mats = debug_info.get("relation_mats", {}) or {}
        rel_rows: List[dict] = []
        for (sid, kw), rel_map in sorted(rel_mats.items(), key=lambda x: (x[0][0], x[0][1])):
            rels = [
                {"from": g, "to": f, "score": float(sc)}
                for (g, f), sc in sorted(rel_map.items(), key=lambda x: (x[0][0], x[0][1]))
            ]
            rel_rows.append(
                {
                    "sentence_id": sid,
                    "keyword": kw,
                    "relations": rels,
                }
            )
        out["relation_mats"] = rel_rows

    return out


# 7) TruthFinder main

def truthfinder_run(
    models: List[str],
    sentence_id: str,
    keywords: List[str],
    results: Dict[str, dict],
    cfg: TruthFinderConfig = TruthFinderConfig(),
    return_debug: bool = False,
) -> Any:
    """
    Default return remains compatible:
      (t_score, s_score, cand_map)
    If return_debug=True:
      (t_score, s_score, cand_map, debug_info)
    """
    objects: List[Tuple[str, str]] = [(sentence_id, kw) for kw in keywords]
    if cfg.support_mode not in ("top1", "soft"):
        raise ValueError(f"Invalid support_mode={cfg.support_mode!r}, expected 'top1' or 'soft'")

    # Raw candidates collected before clustering.
    raw_obj_facts: Dict[Tuple[str, str], List[str]] = {o: [] for o in objects}
    raw_support_hint_by_obj: Dict[Tuple[str, str], Dict[str, float]] = {o: {} for o in objects}
    model_obj_cands: Dict[str, Dict[Tuple[str, str], List[str]]] = {m: {} for m in models}
    raw_top1_choice: Dict[str, Dict[Tuple[str, str], str]] = {m: {} for m in models}

    for m in models:
        payload = results.get(m, {}) or {}
        rows = payload.get("keywords", []) or []

        kw2raw: Dict[str, str] = {}
        for r in rows:
            k = str(r.get("keyword") or "").strip()
            v = str(r.get("meaning_zh") or "").strip()
            if k:
                kw2raw[k] = v

        for kw in keywords:
            o = (sentence_id, kw)
            raw = kw2raw.get(kw, "")
            cands = normalize_meaning_zh_soft(raw, top_n=cfg.topn_candidates)
            cands = [c.strip() for c in (cands or []) if c and str(c).strip()]
            model_obj_cands[m][o] = cands

            if cands:
                raw_top1_choice[m][o] = cands[0]
                for fact in cands:
                    if fact not in raw_obj_facts[o]:
                        raw_obj_facts[o].append(fact)

                ws = _candidate_weights(len(cands), decay=cfg.cand_decay)
                for fact, wgt in zip(cands, ws):
                    raw_support_hint_by_obj[o][fact] = raw_support_hint_by_obj[o].get(fact, 0.0) + float(wgt)

    # Fill empty objects with placeholder and weak support from all models.
    for o in objects:
        if not raw_obj_facts[o]:
            raw_obj_facts[o] = [cfg.empty_fact]
            for m in models:
                model_obj_cands[m][o] = [cfg.empty_fact]
                raw_top1_choice[m][o] = cfg.empty_fact
            raw_support_hint_by_obj[o][cfg.empty_fact] = float(len(models))

    # Per-object clustering.
    obj_facts: Dict[Tuple[str, str], List[str]] = {}
    fact_to_cluster_by_obj: Dict[Tuple[str, str], Dict[str, str]] = {}
    cluster_members_by_obj: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    for o in objects:
        reps, f2c, members = cluster_facts_for_object(
            raw_obj_facts[o],
            cfg,
            support_hint=raw_support_hint_by_obj.get(o, {}),
        )
        if not reps:
            reps = [cfg.empty_fact]
            f2c = {cfg.empty_fact: cfg.empty_fact}
            members = {cfg.empty_fact: [cfg.empty_fact]}
        obj_facts[o] = reps
        fact_to_cluster_by_obj[o] = f2c
        cluster_members_by_obj[o] = members

    # Weighted support over clustered facts.
    support: Dict[Tuple[Tuple[str, str], str], Dict[str, float]] = {}
    top1_choice: Dict[str, Dict[Tuple[str, str], str]] = {m: {} for m in models}

    for m in models:
        for o in objects:
            cands = model_obj_cands.get(m, {}).get(o, [])
            if not cands:
                continue

            if cfg.support_mode == "top1":
                use_cands = cands[:1]
                ws = [1.0] if use_cands else []
            else:
                use_cands = cands
                ws = _candidate_weights(len(cands), decay=cfg.cand_decay)

            for fact, wgt in zip(use_cands, ws):
                rep = fact_to_cluster_by_obj[o].get(fact, fact)
                key = (o, rep)
                support.setdefault(key, {})
                support[key][m] = support[key].get(m, 0.0) + float(wgt)

            top1_raw = raw_top1_choice.get(m, {}).get(o)
            if top1_raw:
                top1_choice[m][o] = fact_to_cluster_by_obj[o].get(top1_raw, top1_raw)

    # Observed dependency + family prior dependency.
    observed_rho = compute_rho_dependency_top1(models, top1_choice)
    dependency = compute_dependency_with_family(models, observed_rho, cfg)

    dep_avg: Dict[str, float] = {}
    for w in models:
        vals = [dependency.get((w, u), 0.0) for u in models if u != w]
        dep_avg[w] = sum(vals) / len(vals) if vals else 0.0

    # Init trust.
    t: Dict[str, float] = {w: float(cfg.t0) for w in models}

    entries: Dict[str, List[Tuple[Tuple[str, str], str, float]]] = {w: [] for w in models}
    for (o, fact), m2w in support.items():
        for w, wgt in m2w.items():
            if wgt > 0:
                entries[w].append((o, fact, float(wgt)))

    s_score: Dict[Tuple[str, str], Dict[str, float]] = {o: {} for o in objects}
    # Use neutral prior so relation terms are effective from the first iteration.
    last_s: Dict[Tuple[str, str], Dict[str, float]] = {
        o: {f: float(cfg.init_last_s) for f in obj_facts[o]}
        for o in objects
    }

    relation_mats: Dict[Tuple[str, str], Dict[Tuple[str, str], float]] = {}
    for o in objects:
        facts = obj_facts[o]
        members = cluster_members_by_obj.get(o)
        if members:
            relation_mats[o] = build_cluster_relation_matrix(facts, members, cfg)
        else:
            relation_mats[o] = build_relation_matrix(facts, cfg)

    t_history: List[Dict[str, float]] = [dict(t)]
    change_history: List[float] = []
    abs_change_history: List[float] = []

    dep_damp = _get_dep_dampening(cfg)

    iter_count = 0
    for _ in range(cfg.max_iter):
        iter_count += 1
        old_t = dict(t)

        tau: Dict[str, float] = {}
        for w in models:
            base = _tau(old_t[w])
            scale = 1.0 - dep_damp * dep_avg.get(w, 0.0)
            scale = max(cfg.min_tau_scale, scale)
            tau[w] = base * scale

        for o in objects:
            facts = obj_facts[o]
            rel_mat = relation_mats.get(o, {})

            base_sigma: Dict[str, float] = {}
            for f in facts:
                m2w = support.get((o, f), {}) or {}
                base_sigma[f] = sum(tau[m] * wgt for m, wgt in m2w.items())

            for f in facts:
                rel_effect = 0.0
                for g in facts:
                    if g == f:
                        continue
                    g_strength = float(last_s.get(o, {}).get(g, 0.0))
                    rel = rel_mat.get((g, f), 0.0)
                    if rel >= 0:
                        rel_effect += cfg.alpha_imp * rel * g_strength
                    else:
                        rel_effect += cfg.alpha_conflict * rel * g_strength

                score = base_sigma[f] + rel_effect
                s_score[o][f] = _sigmoid(cfg.beta * score)

        for o in objects:
            last_s[o] = dict(s_score.get(o, {}))

        new_t: Dict[str, float] = {}
        for w in models:
            es = entries.get(w, [])
            if not es:
                new_t[w] = old_t[w]
                continue

            num = 0.0
            den = 0.0
            for (o, f, wgt) in es:
                num += float(wgt) * float(s_score.get(o, {}).get(f, 0.0))
                den += float(wgt)

            if cfg.use_trust_prior:
                prior = get_trust_prior(w, cfg)
                mu = max(0.0, float(cfg.trust_prior_strength))
                new_value = (mu * prior + num) / (mu + den) if (mu + den) > 0 else prior
            else:
                new_value = (num / den) if den > 0 else old_t[w]

            new_t[w] = min(max(float(new_value), 1e-6), 1 - 1e-6)

        old_vec = [old_t[w] for w in models]
        new_vec = [new_t[w] for w in models]
        cos_sim = _cosine(old_vec, new_vec)
        change = max(0.0, 1.0 - cos_sim)
        max_abs_change = max(abs(new_t[w] - old_t[w]) for w in models) if models else 0.0

        change_history.append(change)
        abs_change_history.append(max_abs_change)
        t = new_t
        t_history.append(dict(t))

        if (
            cfg.early_stop
            and iter_count >= cfg.min_iter
            and change < cfg.delta
            and max_abs_change < cfg.abs_delta
        ):
            break

    cand_map = {o: obj_facts[o] for o in objects}

    if not return_debug:
        return t, s_score, cand_map

    raw_debug_info: Dict[str, Any] = {
        "iter_count": iter_count,
        "t_history": t_history,
        "change_history": change_history,
        "abs_change_history": abs_change_history,
        "dep_avg": dep_avg,
        "support_mode": cfg.support_mode,
        "min_iter": cfg.min_iter,
        "init_last_s": cfg.init_last_s,
        "rho": observed_rho,
        "dependency": dependency,
        "cluster_members": cluster_members_by_obj,
        "support": support,
    }
    if cfg.debug_relations:
        raw_debug_info["relation_mats"] = relation_mats

    debug_info = make_jsonable_debug(raw_debug_info) if cfg.jsonable_debug else raw_debug_info
    return t, s_score, cand_map, debug_info


# 8) Picking truth per keyword

def _pick_from_ranked(
    facts_conf: List[Tuple[str, float]],
    top_k: int,
    margin: float,
) -> List[Tuple[str, float]]:
    if not facts_conf:
        return []
    if len(facts_conf) == 1:
        return facts_conf[:1]

    (f1, c1), (f2, c2) = facts_conf[0], facts_conf[1]
    if float(c1) - float(c2) <= margin:
        return facts_conf[: max(2, top_k)]
    return [(f1, c1)]


def explain_truth_per_keyword(
    sentence_id: str,
    keywords: List[str],
    s_score: Dict[Tuple[str, str], Dict[str, float]],
    support: Optional[Dict[Tuple[Tuple[str, str], str], Dict[str, float]]] = None,
    cluster_members: Optional[Dict[Tuple[str, str], Dict[str, List[str]]]] = None,
    top_k: int = 2,
    margin: float = 0.03,
) -> List[dict]:
    """
    Why multiple truth candidates are allowed:
    in translation, close alternatives can both be valid and uncertainty should be explicit.
    """
    out: List[dict] = []
    for kw in keywords:
        o = (sentence_id, kw)
        facts_conf = list((s_score.get(o, {}) or {}).items())
        facts_conf.sort(key=lambda x: x[1], reverse=True)

        if not facts_conf:
            out.append(
                {
                    "keyword": kw,
                    "truth": [EMPTY_FACT],
                    "conf": [0.0],
                    "candidates": [
                        {
                            "rank": 1,
                            "fact": EMPTY_FACT,
                            "confidence": 0.0,
                            "is_selected": True,
                            "support_weight": 0.0,
                            "support_by_model": {},
                            "cluster_members": [EMPTY_FACT],
                        }
                    ],
                    "cluster_members": {},
                }
            )
            continue

        picked = _pick_from_ranked(facts_conf, top_k=top_k, margin=margin)
        selected_facts = {f for f, _ in picked}
        obj_clusters = cluster_members.get(o, {}) if cluster_members else {}

        candidate_items: List[dict] = []
        for idx, (fact, conf) in enumerate(facts_conf, start=1):
            by_model = (support.get((o, fact), {}) or {}) if support is not None else {}
            cluster_list = obj_clusters.get(fact, [fact])
            item = {
                "rank": idx,
                "fact": fact,
                "confidence": float(conf),
                "is_selected": fact in selected_facts,
                "support_weight": float(sum(by_model.values())),
                "support_by_model": {k: float(v) for k, v in by_model.items()},
                "cluster_members": list(cluster_list),
            }
            candidate_items.append(item)

        row = {
            "keyword": kw,
            "truth": [f for f, _ in picked],
            "conf": [float(c) for _, c in picked],
            "candidates": candidate_items,
            "cluster_members": obj_clusters,
        }
        out.append(row)
    return out


def pick_truth_per_keyword(
    sentence_id: str,
    keywords: List[str],
    s_score: Dict[Tuple[str, str], Dict[str, float]],
    top_k: int = 2,
    margin: float = 0.03,
) -> List[dict]:
    detailed = explain_truth_per_keyword(
        sentence_id=sentence_id,
        keywords=keywords,
        s_score=s_score,
        support=None,
        cluster_members=None,
        top_k=top_k,
        margin=margin,
    )
    return [{"keyword": x["keyword"], "truth": x["truth"], "conf": x["conf"]} for x in detailed]


def rank_translations_by_truth(
    models: List[str],
    results: Dict[str, dict],
    sentence_id: str,
    keywords: List[str],
    truth_rows: List[dict],
    t_score: Dict[str, float],
    *,
    lambda_keyword: float = 0.65,
    lambda_trust: float = 0.25,
    lambda_min: float = 0.10,
) -> List[dict]:
    """
    Rank full-sentence translations using keyword-level truth aggregation outputs.
    This does not change TruthFinder iteration; it only provides downstream reranking.
    """
    kw2truth: Dict[str, List[str]] = {}
    for row in truth_rows or []:
        kw = str(row.get("keyword") or "").strip()
        if not kw:
            continue
        vals = row.get("truth", []) or []
        kw2truth[kw] = [str(x).strip() for x in vals if str(x).strip()]

    ranked: List[dict] = []
    for m in models:
        payload = results.get(m, {}) or {}
        translation = (
            payload.get("translation")
            or payload.get("translation_zh")
            or payload.get("full_translation")
            or ""
        )

        rows = payload.get("keywords", []) or []
        kw2meaning: Dict[str, str] = {}
        for r in rows:
            k = str(r.get("keyword") or "").strip()
            v = str(r.get("meaning_zh") or "").strip()
            if k:
                kw2meaning[k] = v

        keyword_matches: List[dict] = []
        per_kw_scores: List[float] = []

        for kw in keywords:
            truths = kw2truth.get(kw, [])
            raw = kw2meaning.get(kw, "")
            model_candidates = normalize_meaning_zh_soft(raw, top_n=3) or []
            model_candidates = [c.strip() for c in model_candidates if str(c).strip()]

            match = 0.0
            for mc in model_candidates:
                for tv in truths:
                    if mc == tv:
                        match = max(match, 1.0)
                    else:
                        rel = fact_relation_score(mc, tv, TruthFinderConfig())
                        if rel > 0:
                            match = max(match, float(rel))

            per_kw_scores.append(match)
            keyword_matches.append(
                {
                    "keyword": kw,
                    "model_candidates": model_candidates,
                    "truth": truths,
                    "match": float(match),
                }
            )

        keyword_score = (sum(per_kw_scores) / len(per_kw_scores)) if per_kw_scores else 0.0
        min_keyword_score = min(per_kw_scores) if per_kw_scores else 0.0
        trust_score = float(t_score.get(m, 0.0))
        final_score = (
            float(lambda_keyword) * keyword_score
            + float(lambda_trust) * trust_score
            + float(lambda_min) * min_keyword_score
        )

        ranked.append(
            {
                "model": m,
                "translation": str(translation),
                "score": float(final_score),
                "keyword_score": float(keyword_score),
                "trust_score": float(trust_score),
                "min_keyword_score": float(min_keyword_score),
                "keyword_matches": keyword_matches,
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def rank_models_by_trust(t_score: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(t_score.items(), key=lambda x: x[1], reverse=True)