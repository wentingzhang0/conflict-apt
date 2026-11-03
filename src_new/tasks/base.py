"""
TaskAdapter — 任务适配器基类
将任务特化的输入格式化和输出解析从 Agent 中解耦
"""
from abc import ABC, abstractmethod


class TaskAdapter(ABC):
    task_type: str = ""
    task_instruction: str = ""      # 注入 prompt 的任务描述
    answer_format: str = ""         # 告诉 LLM 输出什么格式
    extra_context: str = ""         # 任务附加上下文（如 ATE 的 ATT&CK 技术列表）

    @abstractmethod
    def format_query(self, **kwargs) -> str:
        """将原始输入格式化为 pipeline query 字符串"""
        ...

    @abstractmethod
    def parse_output(self, conclusion: str) -> str:
        """从 pipeline 最终结论中提取任务特化的答案"""
        ...
