import logging
import jsonschema
from typing import Callable, Dict, Any, List
from .schemas import ToolResult, ToolStatus

logger = logging.getLogger(__name__)

class Tool:
    def __init__(self, name: str, description: str, func: Callable[[Dict[str, Any]], ToolResult], parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters
        
    def get_ollama_definition(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def run(self, arguments: Dict[str, Any]) -> ToolResult:
        try:
            jsonschema.validate(instance=arguments, schema=self.parameters)
            return self.func(arguments)
        except jsonschema.exceptions.ValidationError as e:
            logger.warning(f"Tool {self.name} validation failed: {e.message}")
            return ToolResult(status=ToolStatus.ERROR, error=f"Validation Error: Tham số không hợp lệ.")
        except Exception as e:
            logger.exception(f"Internal execution error in Tool {self.name}")
            return ToolResult(
                status=ToolStatus.ERROR, 
                error="Internal tool failure.", 
                metadata={"error_code": "TOOL_INTERNAL_ERROR"}
            )

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools.get(name)

    def get_all_definitions(self) -> List[Dict[str, Any]]:
        return [t.get_ollama_definition() for t in self._tools.values()]

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(status=ToolStatus.ERROR, error=f"Tool {name} không tồn tại trong hệ thống.")
        return tool.run(arguments)

def build_retriever_tool(retriever: Any) -> Tool:
    def retriever_wrapper(args: Dict[str, Any]) -> ToolResult:
        query = args.get("query", "")
        top_k = args.get("top_k", 3)
        
        result = retriever.retrieve(query=query, top_k=top_k)
        status = result.get("metadata", {}).get("status")
        
        if status == "SUCCESS":
            return ToolResult(
                status=ToolStatus.SUCCESS, 
                data={
                    "documents": result.get("documents", []),
                    "metadata": result.get("metadata", {})
                }
            )
        
        status_map = {"NO_RESULTS": ToolStatus.NO_RESULT, "ERROR": ToolStatus.ERROR}
        return ToolResult(
            status=status_map.get(status, ToolStatus.ERROR), 
            error=result.get("metadata", {}).get("error", "Không tìm thấy dữ liệu."),
            metadata={"status": status}
        )

    return Tool(
        name="knowledge_retriever",
        description="Tra cứu tài liệu thô từ cơ sở dữ liệu. Bắt buộc gọi khi cần thông tin thực tế.",
        func=retriever_wrapper,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Câu hỏi cần tra cứu.",
                    "maxLength": 500,
                    "minLength": 1
                },
                "top_k": {
                    "type": "integer",
                    "description": "Số lượng tài liệu cần truy xuất.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 3
                }
            },
            "required": ["query", "top_k"],
            "additionalProperties": False 
        }
    )