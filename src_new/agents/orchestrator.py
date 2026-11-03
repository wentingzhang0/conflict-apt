"""
Orchestrator —— 对外统一的编排入口。

实际编排逻辑在 src.controller.Controller（确定性编排器，创新点 1 控制 + 创新点 3 迭代）。
本类是 Controller 的薄封装：实验脚本 (run_ate/run_mcq/run_rcm) 以 Orchestrator(task, top_k,
num_clusters, max_iterations, eval_samples) 的形式调用，其中只有 task / top_k 传给 Controller；
num_clusters / max_iterations / eval_samples 由 settings.yaml 的 pipeline 配置在内部控制，
CLI 参数仅用于实验日志展示（与 run_taa.py 直接使用 Controller 时的约定一致）。
"""
from src_new.controller import Controller


class Orchestrator:
    """冲突感知多智能体编排器（Controller 的对外别名）。"""

    def __init__(self, task, top_k: int = None,
                 num_clusters: int = None, max_iterations: int = None,
                 eval_samples: int = None, **kwargs):
        # 这些字段保存调用方传入的实验参数，便于日志/调试；
        # 真正的 pipeline 行为由 Controller + settings.yaml 内部配置决定。
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.eval_samples = eval_samples
        self.controller = Controller(task=task, top_k=top_k)

    def analyze(self, query: str):
        """对分析请求 query 执行完整 pipeline，返回 AWM。"""
        return self.controller.analyze(query=query)
