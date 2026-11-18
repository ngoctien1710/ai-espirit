"""
Eval module Tool-use: chạy Agent thật (NativeAgent + AgentExecutor), bắt lại
ĐÚNG chuỗi tool đã gọi (tool_name + arguments) để so với ground truth.

Vấn đề cần giải quyết: AgentExecutor.execute() chỉ trả về {"answer","sources",
"metadata"} khi thành công - KHÔNG trả lại danh sách tool đã gọi. Nhưng eval
Tool-use cần biết chính xác agent đã gọi tool nào, với argument gì.

Giải pháp: bọc ToolRegistry thật bằng 1 lớp "logger" - mỗi lần agent gọi
registry.execute(name, args), logger ghi lại rồi mới chuyển tiếp cho registry
thật xử lý. Không đổi hành vi của agent, chỉ quan sát thêm.

Đặt file này tại: ai-espirit/eval/Pipeline/ (cùng cấp run_full.py)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag.retriever import Retriever
from src.tools.reranker.reranker import Reranker
from src.tools.calculator.calculator import Calculator
from src.tools.datetime.datetime import DateTime
from src.tools.web_search.web_search import WebSearchTool
from src.agent.tools import (
    ToolRegistry,
    build_retriever_tool,
    build_reranker_tool,
    build_calculator_tool,
    build_datetime_tool,
    build_web_search_tool,
)
from src.agent.agent import NativeAgent
from src.agent.executor import AgentExecutor

from eval.dataset_loader import load_dataset
from eval.evaluators import EVALUATOR_REGISTRY
from eval.pipeline import run_pipeline, print_summary

# ------------------- CẤU HÌNH -------------------
INDEX_PATH = str(PROJECT_ROOT / "database" / "index.faiss")
CONTEXTS_PATH = str(PROJECT_ROOT / "database" / "contexts.json")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
AGENT_MODEL = "qwen2.5:3b"
# -------------------------------------------------

# Khởi tạo các module con - CHỈ 1 LẦN
_retriever = Retriever(index_path=INDEX_PATH, contexts_path=CONTEXTS_PATH, model_name=EMBED_MODEL)
_reranker = Reranker()
_calculator = Calculator()
_datetime_tool = DateTime()
_web_search = WebSearchTool()

# Đăng ký đủ 5 tool đúng như hệ thống thật dùng
_base_registry = ToolRegistry()
_base_registry.register(build_retriever_tool(_retriever))
_base_registry.register(build_reranker_tool(_reranker))
_base_registry.register(build_calculator_tool(_calculator))
_base_registry.register(build_datetime_tool(_datetime_tool))
_base_registry.register(build_web_search_tool(_web_search))


class ToolCallLogger:
    """Bọc quanh ToolRegistry thật, ghi lại mọi lệnh gọi tool mà không đổi hành vi."""

    def __init__(self, registry):
        self._registry = registry
        self.calls = []

    def get_all_definitions(self):
        return self._registry.get_all_definitions()

    def get(self, name):
        return self._registry.get(name)

    def execute(self, name, arguments):
        self.calls.append({"tool_name": name, "arguments": arguments})
        return self._registry.execute(name, arguments)


def predict_fn_tool_use(input_data):
    query = input_data["query"]

    logger = ToolCallLogger(_base_registry)
    agent = NativeAgent(registry=logger, model_name=AGENT_MODEL)
    executor = AgentExecutor(agent=agent, max_iterations=5)

    executor.execute(query=query)   # kết quả answer/sources không cần cho eval Tool-use

    return {"tool_calls": logger.calls}


def main():
    dataset = load_dataset("data/tool_use_real.json")
    evaluator = EVALUATOR_REGISTRY["tool_use"]()

    report = run_pipeline(
        dataset=dataset,
        predict_fn=predict_fn_tool_use,
        evaluator=evaluator,
        module_name="tool_use",
        save_path="reports/tool_use_project_report.json",
    )
    print_summary(report)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("Kiểm tra: Ollama đã chạy/pull model chưa? database/index.faiss có tồn tại không?")
