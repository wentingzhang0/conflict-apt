"""
规范键确定性两两冲突匹配（创新点 2，3.2.1–3.2.2）

零 LLM：对一对证据的 ABP canonical_keys 做四维确定性匹配，**每维输出两档之一**：
    confirmed(support / conflict) / none
聚合由 core/conflict.py 完成（任意 conflict→−1；无 conflict、任意 support→+1；全 none→none）。
**不设灰区档**（方案 3.2.1「匹配是判定的（两档），不设灰区档」）：归因/时间线的边界情形
依不对称原则判为 conflict（漏 conflict 代价远高于误报），行为维不同 TTP 不构成冲突故只判 support/none。

规则（方案 3.2.1 表 + 不对称性）：
- attribution（规范组织名 + 别名集）：别名集相交→support；均非空且不相交→conflict；一侧缺→none。
- timeline（规范活跃区间）：真正重叠(共享区间)→support；不相交→conflict；区间不全→none。
- behavior（TTP/工具/目标行业集合）：Jaccard≥上阈→support；否则 none（不同行为不构成冲突）。
- campaign（战役/事件令牌）：令牌重叠→support；否则 none。
attribution/timeline 可 support+conflict；behavior/campaign 只 support/none（3.2.1 不对称性）。
"""
from src_new.utils.settings import get_pipeline_param

_DIM = ["attribution", "timeline", "behavior", "campaign"]


def _embedding_fallback(ck_a: dict, ck_b: dict) -> str:
    """消融 B2：关闭规范键，退化为纯文本嵌入余弦匹配（无归因别名归一等规范化）。

    ck 内的 embedding 字段为 ABP 描述摘要的向量（BGE-M3）。高于阈 → support，
    低于阈 → none；嵌入匹配无法可靠判 conflict（这正是规范键消融要暴露的差距）。"""
    ea = (ck_a or {}).get("embedding") or []
    eb = (ck_b or {}).get("embedding") or []
    if not ea or not eb:
        return "none"
    dot = sum(x * y for x, y in zip(ea, eb))
    na = sum(x * x for x in ea) ** 0.5
    nb = sum(x * x for x in eb) ** 0.5
    if na == 0 or nb == 0:
        return "none"
    thr = get_pipeline_param("canonical_match.embedding_support_threshold", 0.7)
    return "support" if dot / (na * nb) >= thr else "none"


def match_all(ck_a: dict, ck_b: dict) -> dict:
    """逐维匹配，返回 {dim: verdict}，verdict ∈ support|conflict|none（两档，无灰区）。

    消融 B2（canonical_match.use_canonical_keys=false）时退化为嵌入匹配：
    四维全部输出同一嵌入判定（只能 support/none，无法给出 conflict）。"""
    if get_pipeline_param("canonical_match.use_canonical_keys", True) is False:
        v = _embedding_fallback(ck_a, ck_b)
        return {dim: v for dim in _DIM}
    return {
        "attribution": match_attribution(ck_a, ck_b),
        "timeline": match_timeline(ck_a, ck_b),
        "behavior": match_behavior(ck_a, ck_b),
        "campaign": match_campaign(ck_a, ck_b),
    }


def _as_set(x) -> set:
    if x is None:
        return set()
    if isinstance(x, (list, tuple, set)):
        return {str(v).strip().lower() for v in x if str(v).strip()}
    return {str(x).strip().lower()} if str(x).strip() else set()


def _aliases(ck: dict) -> set:
    """归因别名集：canonical_name + aliases 合并、小写化"""
    a = ck.get("attribution") or {}
    names = _as_set(a.get("aliases"))
    cn = a.get("canonical_name")
    if cn:
        names |= _as_set([cn])
    return names


def _timeline_interval(ck: dict):
    """返回 (start, end) 整数年；解析失败返回 (None, None)"""
    t = ck.get("timeline") or {}
    s = _parse_year(t.get("start"))
    e = _parse_year(t.get("end"))
    return s, e


def _parse_year(v):
    if v is None or v == "":
        return None
    s = str(v).strip()
    for tok in s.replace("-", " ").split():
        digits = "".join(ch for ch in tok if ch.isdigit())
        if len(digits) == 4:
            try:
                return int(digits)
            except ValueError:
                continue
    return None


def match_attribution(ck_a: dict, ck_b: dict) -> str:
    sa, sb = _aliases(ck_a), _aliases(ck_b)
    if not sa or not sb:
        return "none"          # 一侧无归因信息，无可比较内容
    if sa & sb:
        return "support"       # 别名集相交 → 同一组织
    # 均非空且不相交 → 明确不同组织 → conflict
    # （疑似子组在抽取阶段尽量归一并入别名集；此处无交集即判 conflict，不对称偏向）
    return "conflict"


def match_timeline(ck_a: dict, ck_b: dict) -> str:
    """真正重叠(共享区间)→support；不相交(含仅相接无共享)→conflict；区间不全→none。
    两档 crisp，不留灰区（方案 3.2.1 表 + 3.1.2 不对称偏向）。"""
    sa, ea = _timeline_interval(ck_a)
    sb, eb = _timeline_interval(ck_b)
    if sa is None or ea is None or sb is None or eb is None:
        return "none"          # 区间信息不全，无可比较内容
    # 不相交（含仅相接而无可共享时段）→ conflict
    if ea < sb or eb < sa:
        return "conflict"
    # 共享区间（含边界同年相接）→ support
    return "support"


def match_behavior(ck_a: dict, ck_b: dict) -> str:
    """Jaccard≥上阈→support；否则 none（不同行为不构成冲突，方案 3.2.1）。"""
    sa = _as_set((ck_a.get("behavior") or {}).get("tokens"))
    sb = _as_set((ck_b.get("behavior") or {}).get("tokens"))
    if not sa or not sb:
        return "none"
    jac = len(sa & sb) / len(sa | sb)
    up = get_pipeline_param("canonical_match.behavior_jaccard_support", 0.5)
    if jac >= up:
        return "support"
    return "none"


def match_campaign(ck_a: dict, ck_b: dict) -> str:
    sa = _as_set((ck_a.get("campaign") or {}).get("tokens"))
    sb = _as_set((ck_b.get("campaign") or {}).get("tokens"))
    if not sa or not sb:
        return "none"
    return "support" if (sa & sb) else "none"


def match_all_deprecated(ck_a: dict, ck_b: dict) -> dict:
    """（已由上方带消融分支的 match_all 取代，保留占位避免外部误引用）"""
    return match_all(ck_a, ck_b)
