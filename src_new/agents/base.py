"""
BaseAgent —— LLM 智能体基类（新版方案，星型拓扑：智能体间不直接通信，经 Controller 中转）
"""


class BaseAgent:
    name: str = "base"

    def log(self, awm, action: str, detail: str = ""):
        awm.log(self.name, action, detail)
