import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# ==========================================
# ⚙️ CẤU HÌNH TÙY CHỈNH (CHỈNH SỬA Ở ĐÂY)
# ==========================================
TEST_DATA_PATH = "dataset/test.json"       # Đường dẫn tới file câu hỏi test
FAISS_INDEX_PATH = "database/index.faiss"  # Đường dẫn tới file vector FAISS
CONTEXTS_PATH = "database/contexts.json"   # Đường dẫn tới file text kho dữ liệu

LOG_SAVE_DIR = "tests/results"             # Thư mục chứa file log kết quả
LOG_FILE_NAME = "test_full_tools_run2.txt" # Tên file log (Tùy ý đổi tên khi chạy mẻ mới)
NUM_SAMPLES = None                         # Số lượng câu mốn test (Để None nếu muốn chạy 100% data)
# ==========================================

# --- Import các thành phần Core ---
from src.agent.schemas import ToolResult, ToolStatus
from src.agent.tools import Tool, ToolRegistry, build_retriever_tool, build_reranker_tool
from src.agent.agent import NativeAgent
from src.agent.executor import AgentExecutor
from eval.metrics import rouge_l_score, compute_bleu_scores

# --- Import toàn bộ Tools ---
from src.tools.retriever.retriever import Retriever
from src.tools.reranker.reranker import Reranker
from src.tools.calculator.calculator import Calculator
from src.tools.datetime.datetime import DateTime
from src.tools.web_search.web_search import WebSearchTool
from src.tools.memory.memory import Memory

# ==========================================
# CÁC HÀM BỌC (WRAPPER) CHO TOOL MỚI
# ==========================================
def build_calculator_tool(calc: Calculator) -> Tool:
    def func(args: Dict[str, Any]) -> ToolResult:
        expr = args.get("expression", "")
        res = calc.calculate(expr)
        if "error" in res:
            return ToolResult(status=ToolStatus.ERROR, error=res["error"])
        return ToolResult(status=ToolStatus.SUCCESS, data=res)
        
    return Tool(
        name="calculator",
        description="Tính toán các biểu thức toán học (cộng, trừ, nhân, chia). Trả về kết quả số.",
        func=func,
        parameters={"type": "object", "properties": {"expression": {"type": "string"}}}
    )

def build_datetime_tool(dt: DateTime) -> Tool:
    def func(args: Dict[str, Any]) -> ToolResult:
        return ToolResult(status=ToolStatus.SUCCESS, data=dt.get_datetime())
        
    return Tool(
        name="datetime",
        description="Lấy ngày, giờ, và thứ hiện tại của hệ thống.",
        func=func,
        parameters={"type": "object", "properties": {}}
    )

def build_web_search_tool(web: WebSearchTool) -> Tool:
    def func(args: Dict[str, Any]) -> ToolResult:
        return web.execute(query=args.get("query", ""), max_results=args.get("max_results", 5))
        
    return Tool(
        name="web_search",
        description="Tìm kiếm thông tin trên internet khi dữ liệu nội bộ không có.",
        func=func,
        parameters={
            "type": "object", 
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5}
            }
        }
    )

# ==========================================
# HÀM CHẠY BENCHMARK CHÍNH
# ==========================================
def run_full_benchmark():
    test_file = ROOT_DIR / TEST_DATA_PATH
    index_path = ROOT_DIR / FAISS_INDEX_PATH
    contexts_path = ROOT_DIR / CONTEXTS_PATH
    
    log_dir = ROOT_DIR / LOG_SAVE_DIR
    log_file = log_dir / LOG_FILE_NAME
    os.makedirs(log_dir, exist_ok=True)
    
    def log_print(message):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"--- BẮT ĐẦU BENCHMARK (FULL TOOLS): {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")

    print(f"Đang khởi tạo hệ thống. Kết quả sẽ được lưu tại: {log_file}")
    
    # 1. Khởi tạo instances của Tools
    retriever = Retriever(index_path=str(index_path), contexts_path=str(contexts_path)) 
    reranker = Reranker(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    calculator = Calculator()
    dt_tool = DateTime()
    web_search = WebSearchTool()
    memory_manager = Memory(mode="retriever_based")
    
    # 2. Đăng ký toàn bộ Tools vào Registry
    registry = ToolRegistry()
    registry.register(build_retriever_tool(retriever))
    registry.register(build_reranker_tool(reranker))
    registry.register(build_calculator_tool(calculator))
    registry.register(build_datetime_tool(dt_tool))
    registry.register(build_web_search_tool(web_search))
    
    # 3. Khởi tạo Agent 
    agent = NativeAgent(registry=registry, model_name="qwen2.5:7b")
    executor = AgentExecutor(agent=agent, max_iterations=6)

    log_print(f"Tải dữ liệu từ {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    total_rouge_f1, total_rouge_recall = 0, 0
    total_bleu = {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}
    
    samples_run = len(dataset) if NUM_SAMPLES is None else min(NUM_SAMPLES, len(dataset))
    
    for i, item in enumerate(dataset[:samples_run], 1):
        print(f"⏳ Đang xử lý: {i}/{samples_run}...", end="\r", flush=True)
        
        question = item.get("question", "")
        ground_truth = str(item.get("answer", ""))
        session_id = f"eval_session_{i}"
        
        # --- MÔ PHỎNG LUỒNG CHẠY CỦA API ---
        mem_search = memory_manager.search(query=question, session_id=session_id, top_k=3)
        history_text = "\n".join([m["content"] for m in mem_search.get("memories", [])])
        
        response = executor.execute(
            query=question,
            session_id=session_id,
            metadata={"route": "knowledge_qa", "history": history_text}
        )
        
        prediction = str(response.get("answer", ""))
        memory_manager.add_turn(session_id, question, prediction)
        
        # --- CHẤM ĐIỂM ---
        sources = response.get("sources", [])
        r_scores = rouge_l_score(prediction, ground_truth)
        b_scores = compute_bleu_scores(prediction, ground_truth)
        
        total_rouge_f1 += r_scores["f1"]
        total_rouge_recall += r_scores["recall"]
        for k in total_bleu:
            total_bleu[k] += b_scores[k]
            
        log_print(f"[{i}/{samples_run}] Hỏi: {question}")
        log_print(f" - Kỳ vọng : {ground_truth}")
        log_print(f" - Thực tế : {prediction}")
        log_print(f" - Điểm    : ROUGE-L(F1)={r_scores['f1']:.4f} | BLEU-1={b_scores['bleu1']:.4f}")
        
        if sources:
            log_print(" - Tài liệu tham chiếu (Retrieved Context):")
            for idx, doc in enumerate(sources):
                content = doc.get("content", "")[:150].replace("\n", " ")
                log_print(f"   [{idx+1}] {content}...")
        log_print("\n")

    log_print("="*50)
    log_print("KẾT QUẢ TRUNG BÌNH (FULL TOOLS):")
    log_print(f"Average ROUGE-L (F1)     : {total_rouge_f1 / samples_run:.4f}")
    for k in total_bleu:
        log_print(f"Average {k.upper():<16} : {total_bleu[k] / samples_run:.4f}")

    print(f"\nHoàn tất! Đã lưu toàn bộ kết quả vào {log_file}.")

if __name__ == "__main__":
    run_full_benchmark()