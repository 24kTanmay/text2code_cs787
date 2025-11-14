from datasets import load_dataset
import json, os

OUTDIR = "data"
os.makedirs(OUTDIR, exist_ok=True)
out_path = os.path.join(OUTDIR, "python_data.json")

print("Opening dataset in streaming mode (no full download into memory)...")
ds = load_dataset("huawei-noah/python_text2code", split="train", streaming=True)

count = 0
with open(out_path, "w", encoding="utf-8") as f:
    for row in ds:            # iterates over the dataset stream
        rec = {
            "docstring": row.get("docstring", ""),
            "signature": row.get("signature", ""),
            "code": row.get("code", ""),
            "_id": str(count)
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        count += 1
        if count % 500000 == 0:
            print(f"Written {count} records...")
print("Wrote", count, "records to", out_path)
