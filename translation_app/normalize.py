"""
Soft normalization for Chinese glosses (meaning_zh) produced by LLMs.

Design goals:
1) Preserve differences (avoid over-aggressive canonicalization).
2) Split multi-gloss outputs into candidates.
3) Remove obvious noise/templates while keeping the core meaning.
4) Detect and down-rank garbage outputs (esp. prompt leakage / gibberish).
5) Provide stable, controllable output for Truth Discovery.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast


# Config (tunable)
@dataclass(frozen=True)
class SoftNormConfig:
    top_n: int = 3

    # Split rules: keep it conservative (do not split on "和/与/及" to avoid breaking phrases)
    split_regex: str = r"[，,;；/｜|、\n]+|(?:\s{2,})|(?:\s*/\s*)|(?:\s*\|\s*)"

    # Remove bracketed POS/notes
    remove_parentheses: bool = True

    # Translation candidates should stay short; long explanatory sentences should usually be filtered.
    # Raised slightly for technical/security phrases without allowing long sentence-like explanations.
    max_candidate_len: int = 20

    # Minimum length (too short like "的" or "了" is meaningless)
    min_candidate_len: int = 1

    # If candidate has too many non-CJK characters, treat as low quality
    non_cjk_ratio_bad: float = 0.65

    # Garbage detection thresholds
    max_punct_ratio: float = 0.35
    max_repeat_char_run: int = 4

    # Default to quality-first ordering so TruthFinder support_mode="top1" receives the best top1 candidate.
    preserve_order: bool = False

    # Default threshold should block weak explanation fragments while keeping short technical terms.
    min_quality_score: float = 0.0

    # Use a stricter threshold when the whole raw output looks like prompt leakage / garbage.
    min_quality_score_when_raw_garbage: float = 0.5

    # If any of these patterns appear, it's probably prompt leakage / system message
    hard_blacklist_patterns: Tuple[str, ...] = (
        r"只输出\s*JSON",
        r"不要输出.*多余文字",
        r"关键词数组|keywords\s*数组|输入关键词",
        r"输出格式|严格如下|示例：|正确输出",
        r"```json|```",
        r"\bjson\b",
        r"\bkeywords?\b|\bmeanings?\b",
        r'"keywords"\s*:|"meanings"\s*:',
        r"\b(system|assistant|user)\b",
        r"role\s*[:：]\s*",
        r"temperature|top_p|num_predict",
    )

    # Light alias: ONLY near-certain equivalence (keep this tiny)
    alias_map: Optional[Dict[str, str]] = None


DEFAULT_ALIAS = {
    "站点": "网站",
    "网页": "网站",
    "網站": "网站",
}

DEFAULT_CFG = SoftNormConfig(alias_map=DEFAULT_ALIAS)


# Regex / helpers
_PARENS_RE = re.compile(r"[\(\（].*?[\)\）]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_KEEP_CORE_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff#._\-\+/：:·]+")

# Remove common leading templates (keep conservative!)
_PREFIX_RE = re.compile(
    r"^(?:"
    r"指的是|指为|即|也就是|表示|意为|意思是|释义是|翻译为|译为|用于|用来|"
    r"中文意思是|中文释义是|该词意为|该词表示|"
    r"该关键词表示|在这里表示|在此处表示|在上下文中表示|在本句中表示|在当前语境中表示|"
    r"可理解为|可以理解为|可以译作|可译为|常译为|通常译为"
    r")\s*[:：]?\s*"
)

# Remove common trailing fillers
_SUFFIX_RE = re.compile(r"(等|等等|之类|相关|方面)\s*$")

# Basic char classes
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_PUNCT_RE = re.compile(r"[，,;；:：。.!！?？、/｜|\-—_~`'\"“”‘’()\[\]{}<>]")
_EN_RE = re.compile(r"[A-Za-z]")
_TECH_TOKEN_RE = re.compile(
    r"^(?:"
    r"C\+\+|C#|\.NET|"
    r"CVE-\d{4}-\d{3,}|"
    r"[A-Z]{2,}(?:[0-9]+)?|"
    r"[A-Za-z0-9.#_+-]+(?:[:/.-][A-Za-z0-9.#_+-]+)+"
    r")$"
)


def format_candidates(cands: List[str], sep: str = " / ") -> str:
    return sep.join(cands) if cands else "(空)"


# Garbage / quality scoring
def _has_hard_blacklist(text: str, cfg: SoftNormConfig) -> bool:
    for pat in cfg.hard_blacklist_patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


def _repeat_run_too_long(text: str, max_run: int) -> bool:
    # e.g., "哈哈哈哈哈", "aaaaaa", "。。。" etc.
    if not text:
        return False
    run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            run += 1
            if run > max_run:
                return True
        else:
            run = 1
    return False


def _punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    p = len(_PUNCT_RE.findall(text))
    return p / max(1, len(text))


def _non_cjk_ratio(text: str) -> float:
    if not text:
        return 1.0
    cjk = len(_CJK_RE.findall(text))
    return 1.0 - (cjk / max(1, len(text)))


def is_garbage_gloss(raw: str, cfg: SoftNormConfig = DEFAULT_CFG) -> bool:
    """
    Decide if the whole raw output is obviously garbage/prompt leakage.
    If True, caller may choose to return [] or keep only weak candidates.
    """
    if not raw or not raw.strip():
        return True

    t = unicodedata.normalize("NFKC", raw.strip())

    # prompt leakage: strongest signal
    if _has_hard_blacklist(t, cfg):
        return True

    # extreme punctuation or repeated runs
    if _punct_ratio(t) > cfg.max_punct_ratio:
        return True

    if _repeat_run_too_long(t, cfg.max_repeat_char_run):
        return True

    # mostly non-CJK and not a simple English word (we expect Chinese gloss)
    if _non_cjk_ratio(t) > cfg.non_cjk_ratio_bad and not _EN_RE.search(t):
        return True

    return False


def _candidate_quality(cand: str, cfg: SoftNormConfig) -> float:
    """
    Higher is better. Used for sorting candidates.
    We DON'T want to overfit; just stable heuristics:
      - prefer having CJK
      - prefer shorter (like gloss), but not too short
      - penalize too many punct / too many non-CJK
      - penalize if looks like a sentence / instruction
    """
    if not cand:
        return -1e9

    L = len(cand)
    cjk = len(_CJK_RE.findall(cand))
    non_cjk = L - cjk
    punct_r = _punct_ratio(cand)
    non_cjk_r = non_cjk / max(1, L)
    looks_technical = _looks_like_technical_term(cand)

    score = 0.0

    # Prefer Chinese glosses, but keep common technical/security terms viable.
    if cjk > 0:
        score += 2.0
    elif looks_technical:
        score += 1.2
    else:
        score -= 1.5

    # Length preference: gloss-like often 2~8 chars
    if 2 <= L <= 8:
        score += 1.5
    elif 1 <= L <= cfg.max_candidate_len:
        score += 0.8
    else:
        score -= 2.0  # too long

    # Penalize punctuation / non-CJK ratio, but be gentler for structured technical terms.
    if looks_technical:
        score -= 0.8 * punct_r
        score -= 0.3 * non_cjk_r
    else:
        score -= 2.5 * punct_r
        score -= 1.8 * non_cjk_r

    # Penalize if looks like prompt leakage even in candidate
    if _has_hard_blacklist(cand, cfg):
        score -= 5.0

    # Penalize repeated run
    if _repeat_run_too_long(cand, cfg.max_repeat_char_run):
        score -= 2.0

    return score


def _looks_like_technical_term(cand: str) -> bool:
    if not cand:
        return False
    if _TECH_TOKEN_RE.fullmatch(cand):
        return True
    if (
        _CJK_RE.search(cand)
        and _EN_RE.search(cand)
        and not re.search(r"\s", cand)
        and not re.search(r"[,，;；|]", cand)
        and "/" not in cand
    ):
        return True
    return False


def _candidate_reason_tags(
    cand: str,
    score: float,
    threshold: float,
    raw_is_garbage: bool,
    cfg: SoftNormConfig,
) -> List[str]:
    reasons: List[str] = []
    if _CJK_RE.search(cand):
        reasons.append("contains_cjk")
    if _looks_like_technical_term(cand):
        reasons.append("technical_term")
    if 2 <= len(cand) <= 8:
        reasons.append("gloss_length")
    elif len(cand) <= cfg.max_candidate_len:
        reasons.append("within_max_len")
    if not _has_hard_blacklist(cand, cfg):
        reasons.append("not_blacklisted")
    if raw_is_garbage:
        reasons.append("raw_is_garbage")
    if score >= threshold:
        reasons.append("passes_quality_threshold")
    else:
        reasons.append("below_quality_threshold")
    return reasons


# Cleaning & splitting
def _clean_piece(piece: str, cfg: SoftNormConfig) -> str:
    s = unicodedata.normalize("NFKC", (piece or "").strip())
    if not s:
        return ""

    if cfg.remove_parentheses:
        s = _PARENS_RE.sub("", s).strip()

    s = _PREFIX_RE.sub("", s).strip()
    s = _SUFFIX_RE.sub("", s).strip()

    # collapse spaces
    s = _MULTI_SPACE_RE.sub(" ", s).strip()

    # Strip common edge punctuation, but keep leading "." for terms like ".NET".
    s = s.strip(" \"'“”‘’")
    s = re.sub(r"^[。!！?？:：·]+", "", s)
    s = re.sub(r"[。.!！?？:：·]+$", "", s)

    # Keep core chars plus technical/security symbols to avoid breaking terms like
    # C++, C#, .NET, Node.js, CVE-2024-1234, XSS/CSRF, OAuth2.0.
    s = _KEEP_CORE_RE.sub("", s).strip()

    # alias (tiny!)
    if cfg.alias_map and s in cfg.alias_map:
        s = cfg.alias_map[s]

    return s


def _split_candidates(raw: str, cfg: SoftNormConfig) -> List[str]:
    """
    Split raw into candidate pieces conservatively.
    Avoid splitting by "和/与/及" because it may break correct phrases.
    """
    t = unicodedata.normalize("NFKC", (raw or "").strip())
    if not t:
        return []
    if _looks_like_technical_term(t):
        return [t]
    splitter = re.compile(cfg.split_regex)
    parts = [p.strip() for p in splitter.split(t) if p and p.strip()]
    return parts if parts else [t]


def _build_debug_payload(
    raw: str,
    top_n: Optional[int],
    cfg: SoftNormConfig,
) -> Dict[str, object]:
    if top_n is None:
        top_n = cfg.top_n
    top_n = max(1, int(top_n))

    if not raw or not str(raw).strip():
        return {
            "raw": raw,
            "raw_is_garbage": True,
            "split_parts": [],
            "cleaned_candidates": [],
            "quality_scores": {},
            "final_candidates": [],
        }

    text = str(raw).strip()
    raw_is_garbage = is_garbage_gloss(text, cfg)
    parts = _split_candidates(text, cfg)

    cleaned_candidates: List[str] = []
    candidate_details: List[Dict[str, object]] = []
    seen = set()
    threshold = (
        cfg.min_quality_score_when_raw_garbage
        if raw_is_garbage
        else cfg.min_quality_score
    )

    for part in parts:
        cand = _clean_piece(part, cfg)
        detail: Dict[str, object] = {
            "source": part,
            "candidate": cand,
            "quality_score": None,
            "kept": False,
            "reason": [],
        }
        if not cand:
            detail["reason"] = ["empty_after_clean"]
            candidate_details.append(detail)
            continue

        if len(cand) < cfg.min_candidate_len:
            detail["reason"] = ["too_short"]
            candidate_details.append(detail)
            continue

        if len(cand) > cfg.max_candidate_len:
            detail["reason"] = ["too_long"]
            candidate_details.append(detail)
            continue

        if raw_is_garbage:
            if not _CJK_RE.search(cand) and not _looks_like_technical_term(cand):
                detail["reason"] = ["raw_is_garbage", "no_cjk_or_technical_signal"]
                candidate_details.append(detail)
                continue
            if _has_hard_blacklist(cand, cfg):
                detail["reason"] = ["raw_is_garbage", "hard_blacklist"]
                candidate_details.append(detail)
                continue
            if _punct_ratio(cand) > 0.25:
                detail["reason"] = ["raw_is_garbage", "punct_ratio_too_high"]
                candidate_details.append(detail)
                continue
            if _non_cjk_ratio(cand) > cfg.non_cjk_ratio_bad and not _looks_like_technical_term(cand):
                detail["reason"] = ["raw_is_garbage", "non_cjk_ratio_too_high"]
                candidate_details.append(detail)
                continue

        score = _candidate_quality(cand, cfg)
        detail["quality_score"] = score
        detail["reason"] = _candidate_reason_tags(cand, score, threshold, raw_is_garbage, cfg)

        if score < threshold:
            candidate_details.append(detail)
            continue

        if cand not in seen:
            seen.add(cand)
            cleaned_candidates.append(cand)
            detail["kept"] = True
            candidate_details.append(detail)
        else:
            detail["reason"] = list(detail["reason"]) + ["duplicate"]
            candidate_details.append(detail)

    quality_scores = {cand: _candidate_quality(cand, cfg) for cand in cleaned_candidates}
    kept_candidates = list(cleaned_candidates)

    if cfg.preserve_order:
        final_candidates = kept_candidates[:top_n]
    else:
        final_candidates = sorted(
            kept_candidates,
            key=lambda cand: quality_scores[cand],
            reverse=True,
        )[:top_n]

    return {
        "raw": raw,
        "raw_is_garbage": raw_is_garbage,
        "split_parts": parts,
        "cleaned_candidates": cleaned_candidates,
        "candidate_details": candidate_details,
        "quality_scores": quality_scores,
        "final_candidates": final_candidates,
    }


# Main API
def normalize_meaning_zh_soft(
    raw: str,
    top_n: Optional[int] = None,
    cfg: SoftNormConfig = DEFAULT_CFG,
) -> List[str]:
    """
    Soft-normalize a Chinese gloss string into up to top_n candidate facts.
    By default, preserve the model's candidate order after filtering/cleaning.
    """
    payload = _build_debug_payload(raw, top_n, cfg)
    return cast(List[str], payload["final_candidates"])


def debug_normalize(
    raw: str,
    top_n: Optional[int] = None,
    cfg: SoftNormConfig = DEFAULT_CFG,
) -> Dict[str, object]:
    """
    Return intermediate normalization steps for debugging.
    """
    return _build_debug_payload(raw, top_n, cfg)
