# TC-Agent

TC-Agent là một hệ thống AI Agent phục vụ các tác vụ liên quan đến **thờ cúng và văn hóa Việt Nam**.

Dự án được xây dựng theo kiến trúc module, trong đó Agent có thể sử dụng các công cụ như **RAG, Reranker và Memory**.

## Cấu trúc thư mục

```text
tc-agent/
│
├── src/
│   ├── agent/                  # Code Agent (ReAct, prompt, reasoning)
│   │
│   ├── tools/                  # Các tool mà Agent có thể sử dụng
│   │   ├── rag/                # Code RAG (Load data, chunking, retrieval)
│   │   ├── reranker/           # Code Reranker (Cross-Encoder, lọc lại top K)
│   │   └── memory/             # Code Memory (Lưu lịch sử chat, session)
│   │
│   ├── router/                 # Router (Phân loại query, chặn OOD)
│   │
│   └── main.py                 # File chạy thử, nối tất cả lại với nhau
│
├── eval/                       # Code chấm điểm, đo đạc metric (Recall, F1...)
│
└── requirements.txt            # Thư viện cần cài
```

## Môi trường

* **Python 3.11**
* Các thư viện cần thiết được liệt kê trong `requirements.txt`.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

