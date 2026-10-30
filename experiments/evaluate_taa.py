"""
CTIBench-TAA 官方评估逻辑
BFS 别名/关联组匹配，来自 maveryn/cti-bench evaluation.ipynb
"""
import os
import pickle
from collections import deque

# ── 加载别名/关联字典 ────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PICKLE_DIR = os.path.join(_HERE, "..", "data", "cti-bench-official")


def _load_dict(name: str) -> dict:
    path = os.path.join(_PICKLE_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)


_alias_dict = None
_related_dict = None


def _ensure_dicts():
    global _alias_dict, _related_dict
    if _alias_dict is None:
        _alias_dict = _load_dict("alias_dict.pickle")
        _related_dict = _load_dict("related_dict.pickle")


def _normalize_dicts(alias_dict, related_dict):
    """小写 + 双向连接"""
    alias_dict = {
        k.strip().lower(): [v.strip().lower() for v in val]
        for k, val in alias_dict.items()
    }
    for actor in list(alias_dict):
        for alias in alias_dict[actor]:
            if actor not in alias_dict.setdefault(alias, []):
                alias_dict[alias].append(actor)

    related_dict = {
        k.strip().lower(): [v.strip().lower() for v in val]
        for k, val in related_dict.items()
    }
    for actor in list(related_dict):
        for related_actor in related_dict[actor]:
            if actor not in related_dict.setdefault(related_actor, []):
                related_dict[related_actor].append(actor)

    return alias_dict, related_dict


def is_alias_connected(actor1, actor2, alias_dict):
    """BFS 沿别名链判断是否连通"""
    visited = set()
    queue = deque([actor1])
    while queue:
        node = queue.popleft()
        if node == actor2:
            return True
        if node in visited:
            continue
        visited.add(node)
        for neighbor in alias_dict.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
    return False


def is_related_connected(actor1, actor2, alias_dict, related_dict):
    """BFS 沿别名 + 关联组判断是否连通"""
    visited = set()
    queue = deque([actor1])
    while queue:
        node = queue.popleft()
        if node == actor2:
            return True
        if node in visited:
            continue
        visited.add(node)
        for neighbor in alias_dict.get(node, []) + related_dict.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
    return False


def threat_actor_connection(gt: str, pred: str) -> str:
    """
    判断预测与 ground truth 的关系
    Returns:
        "C" - Correct (别名链连通)
        "P" - Plausible (关联组链连通)
        "I" - Incorrect
    """
    _ensure_dicts()
    a, r = _normalize_dicts(_alias_dict, _related_dict)

    gt_lower = gt.strip().lower()
    pred_lower = pred.strip().lower()

    if is_alias_connected(gt_lower, pred_lower, a):
        return "C"
    if is_related_connected(gt_lower, pred_lower, a, r):
        return "P"
    return "I"


def compute_taa_accuracy(ground_truths: list[str], predictions: list[str]):
    """
    计算 CTIBench-TAA 的 Correct Accuracy 和 Plausible Accuracy
    Args:
        ground_truths: 50 个 ground truth 标签
        predictions: 50 个模型预测结果
    Returns:
        (correct_acc, plausible_acc) 百分比
    """
    assert len(ground_truths) == len(predictions)
    correct = 0
    plausible = 0
    details = []

    for gt, pred in zip(ground_truths, predictions):
        res = threat_actor_connection(gt, pred)
        details.append({"gt": gt, "pred": pred, "result": res})
        if res == "C":
            correct += 1
        elif res == "P":
            plausible += 1

    total = len(ground_truths)
    return correct / total * 100, (correct + plausible) / total * 100, details
