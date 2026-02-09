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
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


# Config (tunable)
@dataclass(frozen=True)
class SoftNormConfig:
    top_n: int = 3

    # Split rules: keep it conservative (do not split on "和/与/及" to avoid breaking phrases)
    split_regex: str = r"[，,;；/｜|、\n]+|(?:\s{2,})|(?:\s*/\s*)|(?:\s*\|\s*)"

    # Remove bracketed POS/notes
    remove_parentheses: bool = True

    # Maximum candidate length to keep as a "gloss" (long sentences are likely noise)
    max_candidate_len: int = 16

    # Minimum length (too short like "的" or "了" is meaningless)
    min_candidate_len: int = 1

    # If candidate has too many non-CJK characters, treat as low quality
    non_cjk_ratio_bad: float = 0.65

    # Garbage detection thresholds
    max_punct_ratio: float = 0.35
    max_repeat_char_run: int = 4

    # If any of these patterns appear, it's probably prompt leakage / system message
    hard_blacklist_patterns: Tuple[str, ...] = (
        r"只输出\s*JSON",
        r"不要输出.*多余文字",
        r"关键词数组|keywords\s*数组|输入关键词",
        r"输出格式|严格如下|示例：|正确输出",
        r"```json|```",
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
_KEEP_CORE_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff\-\+·]+")

# Remove common leading templates (keep conservative!)
_PREFIX_RE = re.compile(
    r"^(指的是|指为|即|也就是|表示|意为|意思是|释义是|翻译为|译为|用于|用来)\s*[:：]?\s*"
)

# Remove common trailing fillers
_SUFFIX_RE = re.compile(r"(等|等等|之类|相关|方面)\s*$")

# Basic char classes
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_PUNCT_RE = re.compile(r"[，,;；:：。.!！?？、/｜|\-—_~`'\"“”‘’()\[\]{}<>]")
_EN_RE = re.compile(r"[A-Za-z]")


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
        if text[i] == text[i-1]:
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

    t = raw.strip()

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

    score = 0.0

    # CJK presence is crucial
    score += 2.0 if cjk > 0 else -1.5

    # Length preference: gloss-like often 2~8 chars
    if 2 <= L <= 8:
        score += 1.5
    elif 1 <= L <= cfg.max_candidate_len:
        score += 0.8
    else:
        score -= 2.0  # too long

    # Penalize punctuation / non-CJK ratio
    score -= 2.5 * punct_r
    score -= 1.8 * non_cjk_r

    # Penalize if looks like prompt leakage even in candidate
    if _has_hard_blacklist(cand, cfg):
        score -= 5.0

    # Penalize repeated run
    if _repeat_run_too_long(cand, cfg.max_repeat_char_run):
        score -= 2.0

    return score

# Cleaning & splitting
def _clean_piece(piece: str, cfg: SoftNormConfig) -> str:
    s = (piece or "").strip()
    if not s:
        return ""

    if cfg.remove_parentheses:
        s = _PARENS_RE.sub("", s).strip()

    s = _PREFIX_RE.sub("", s).strip()
    s = _SUFFIX_RE.sub("", s).strip()

    # collapse spaces
    s = _MULTI_SPACE_RE.sub(" ", s).strip()

    # strip quotes / ends
    s = s.strip(" \"'“”‘’。.!！?？:：·")

    # keep only core chars, but allow - + · for things like "非线性-回归"
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
    t = (raw or "").strip()
    if not t:
        return []
    splitter = re.compile(cfg.split_regex)
    parts = [p.strip() for p in splitter.split(t) if p and p.strip()]
    return parts if parts else [t]

# Main API
def normalize_meaning_zh_soft(raw: str, top_n: Optional[int] = None, cfg: SoftNormConfig = DEFAULT_CFG) -> List[str]:
    """
    Soft-normalize a Chinese gloss string into up to top_n candidate facts.
    Returns candidates in descending quality order.
    """
    if top_n is None:
        top_n = cfg.top_n
    top_n = max(1, int(top_n))

    if not raw or not str(raw).strip():
        return []

    text = str(raw).strip()

    # If the whole output is garbage (prompt leakage etc.), still try salvage
    # but we will be very strict on candidates.
    raw_is_garbage = is_garbage_gloss(text, cfg)

    parts = _split_candidates(text, cfg)

    cands: List[str] = []
    seen = set()

    for p in parts:
        c = _clean_piece(p, cfg)
        if not c:
            continue

        # basic length checks
        if len(c) < cfg.min_candidate_len:
            continue

        # If too long, it's likely a sentence or noisy explanation; drop.
        # (You can relax this if you expect phrase-level glosses.)
        if len(c) > cfg.max_candidate_len:
            continue

        # If raw is garbage, be stricter:
        if raw_is_garbage:
            # must contain some CJK to be considered a valid Chinese gloss
            if not _CJK_RE.search(c):
                continue
            # must not contain hard blacklist patterns
            if _has_hard_blacklist(c, cfg):
                continue
            # punctuation ratio should be low
            if _punct_ratio(c) > 0.25:
                continue

        if c not in seen:
            seen.add(c)
            cands.append(c)

    if not cands:
        return []

    # Sort by heuristic quality (descending)
    cands.sort(key=lambda x: _candidate_quality(x, cfg), reverse=True)

    return cands[:top_n]
