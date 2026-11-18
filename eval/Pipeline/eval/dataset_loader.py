"""
Đọc dataset test case từ file JSON.

Format file JSON kỳ vọng (list các test case):
[
    {
        "id": "case_001",
        "case_type": "normal" | "edge" | "adversarial",
        "input": { ... theo đúng Request schema của module  },
        "ground_truth": { ... }
    },
    ...
]
"""

import json


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Dataset {path} phải là 1 list các test case.")

    for i, case in enumerate(data):
        if "input" not in case or "ground_truth" not in case:
            raise ValueError(
                f"Test case thứ {i} trong {path} thiếu field 'input' hoặc 'ground_truth'."
            )
        case.setdefault("id", f"case_{i:04d}")
        case.setdefault("case_type", "normal")

    return data


def load_dataset_by_module(path, module_name):
    """
    Đọc 1 file JSON GỘP nhiều module (mỗi case có thêm field "module"), chỉ lấy
    ra các case thuộc đúng module_name. File gộp có dạng:

    [
        {"module": "retriever", "id": "...", "case_type": "normal", "input": {...}, "ground_truth": {...}},
        {"module": "rag", "id": "...", ...},
        ...
    ]

    file router_sample.json/retriever_sample.json/...
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Dataset {path} phải là 1 list các test case.")

    filtered = []
    for i, case in enumerate(data):
        if case.get("module") != module_name:
            continue
        if "input" not in case or "ground_truth" not in case:
            raise ValueError(
                f"Case thứ {i} (module={module_name}) trong {path} thiếu field 'input' hoặc 'ground_truth'."
            )
        case = dict(case)  # tránh sửa nhầm dict gốc
        case.pop("module", None)
        case.setdefault("id", f"{module_name}_{i:04d}")
        case.setdefault("case_type", "normal")
        filtered.append(case)

    if not filtered:
        raise ValueError(f"Không tìm thấy case nào có module='{module_name}' trong {path}.")

    return filtered


def list_modules_in_dataset(path):
    """Liệt kê các module có trong 1 file dataset gộp, kèm số lượng case mỗi module."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = {}
    for case in data:
        m = case.get("module", "(không có field 'module')")
        counts[m] = counts.get(m, 0) + 1
    return counts
