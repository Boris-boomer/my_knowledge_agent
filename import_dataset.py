import json
import os

INPUT_FILE = "rag_instruct_test_dataset_0.jsonl"
OUTPUT_DIR = "data/rag_instruct"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        try:
            record = json.loads(line)
            context = record.get("context", "").strip()
            if not context:
                continue
            with open(os.path.join(OUTPUT_DIR, f"sample_{idx+1:04d}.txt"), "w", encoding="utf-8") as out:
                out.write(context)
            print(f"已写入 sample_{idx+1:04d}.txt")
        except Exception as e:
            print(f"第 {idx+1} 行解析失败: {e}")

print("导入完成！")