import sys
from pathlib import Path

# eval/Pipeline/run_full.py -> lùi 2 cấp ra tới thư mục gốc ai-espirit/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # để import được package "eval" của pipeline

# ------------------- import module THẬT từ src/ (đường dẫn đã sửa đúng) -------------------
from src.rag.retriever import Retriever
from src.rag.generator import OllamaGenerator
from src.query_router.router_interface import QueryRouter
from src.tools.reranker.reranker import Reranker
from src.tools.calculator.calculator import Calculator

from eval.dataset_loader import load_dataset_by_module
from eval.evaluators import EVALUATOR_REGISTRY
from eval.pipeline import run_pipeline, print_summary

# ------------------- CẤU HÌNH -------------------
INDEX_PATH = str(PROJECT_ROOT / "database" / "index.faiss")
CONTEXTS_PATH = str(PROJECT_ROOT / "database" / "contexts.json")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "qwen2.5:3b"
ROUTER_MODEL = "qwen2.5:7b"
ALL_DATA_PATH = "data/all_data.json"   # file dataset đã gộp
# -------------------------------------------------

retriever = Retriever(index_path=INDEX_PATH, contexts_path=CONTEXTS_PATH, model_name=EMBED_MODEL)
generator = OllamaGenerator(model_name=GEN_MODEL)
router = QueryRouter(backend="llm", model_name=ROUTER_MODEL)
reranker = Reranker()
calculator = Calculator()


# ---------- Retriever ----------

def predict_fn_retriever(input_data):
    query = input_data["query"]
    top_k = input_data.get("top_k", 5)
    return retriever.retrieve(query, top_k=top_k)


# ---------- RAG (Retriever -> Generator) ----------

def predict_fn_rag(input_data):
    query = input_data["query"]
    retrieval_res = retriever.retrieve(query, top_k=5)

    if retrieval_res.get("metadata", {}).get("status") != "SUCCESS":
        return {"answer": "", "sources": [], "metadata": {"status": retrieval_res.get("metadata", {}).get("status")}}

    gen_result = generator.generate(query, retrieval_res["documents"])

    if gen_result["status"] != "SUCCESS":
        return {"answer": "", "sources": [], "metadata": {"status": gen_result["status"], "error": gen_result.get("error")}}

    sources = [
        {
            "document_id": c.get("chunk_id") or c.get("source", "unknown"),
            "content": c.get("content", ""),
            "metadata": {"source": c.get("source", "")},
        }
        for c in gen_result.get("citations", [])
    ]

    return {"answer": gen_result["answer"], "sources": sources, "metadata": {"status": "SUCCESS"}}


# ---------- Router ----------
# LƯU Ý: router thật trả 5 nhãn (knowledge_qa/product_search/multi_step/ambiguous/ood).
# data/all_data.json (case module="router") đã dùng đúng 5 nhãn này.

def predict_fn_router(input_data):
    query = input_data["query"]
    decision = router.route(query)   # trả về RouterDecision (pydantic model)
    return {
        "route": decision.route,
        "confidence": decision.confidence,
        "reason": None,
    }


# ---------- Reranker ----------
# LƯU Ý: reranker cần "documents" (danh sách candidate) trong input, không chỉ query.
# Dataset case module="reranker" trong all_data.json phải có sẵn candidate documents.

def predict_fn_reranker(input_data):
    query = input_data["query"]
    documents = input_data["documents"]
    top_k = input_data.get("top_k", 5)
    return reranker.rerank(query=query, documents=documents, top_k=top_k)


# ---------- Calculator ----------
# LƯU Ý: Calculator chỉ hiểu BIỂU THỨC SỐ HỌC THUẦN (vd "125 + 37"), không hiểu
# câu hỏi tiếng Việt. Case module="calculation" cần field "expression".

def predict_fn_calculation(input_data):
    expression = input_data["expression"]
    result = calculator.calculate(expression)
    if "error" in result:
        return {"result": None, "error": result["error"]}
    return {"result": result["result"]}


# ---------------------------------------------------------------------

def main():
    jobs = [
        ("retriever", predict_fn_retriever, "reports/retriever_project_report.json"),
        ("rag", predict_fn_rag, "reports/rag_project_report.json"),
        ("router", predict_fn_router, "reports/router_project_report.json"),
        ("reranker", predict_fn_reranker, "reports/reranker_project_report.json"),
        ("calculation", predict_fn_calculation, "reports/calculation_project_report.json"),
    ]

    for module_name, predict_fn, save_path in jobs:
        try:
            dataset = load_dataset_by_module(ALL_DATA_PATH, module_name)
        except ValueError as e:
            print(f"⏭️  Bỏ qua module '{module_name}': {e}")
            print()
            continue

        print(f"--- Đang chạy module: {module_name} ({len(dataset)} case) ---")
        evaluator = EVALUATOR_REGISTRY[module_name]()

        report = run_pipeline(
            dataset=dataset,
            predict_fn=predict_fn,
            evaluator=evaluator,
            module_name=module_name,
            save_path=save_path,
        )

        print_summary(report)
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("Kiểm tra: Ollama đã chạy/pull đủ model chưa? database/index.faiss có tồn tại không?")