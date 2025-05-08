from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# 1) Load full Spider train split
dataset = load_dataset("xlangai/spider", split="train")

# 2) Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True
)
print("model loaded")

def generate_sql(prompt: str) -> str:
    inputs  = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs, max_length=1000, temperature=0.2, do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# 3) Loop over every example, include NL question as its own column
results = []
total = len(dataset)
for idx, ex in enumerate(dataset):
    instance_id = ex.get("query_id", idx)
    nl_query    = ex["question"]
    db_id       = ex["db_id"]

    prompt = (
        f"Generate an SQL query using the database name {db_id} "
        f"for the following question:\n\n{nl_query}"
    )
    sql = generate_sql(prompt)

    results.append({
        "instance_id":   instance_id,
        "nl_query":      nl_query,
        "generated_sql": sql
    })

    # progress logging every 100
    if (idx + 1) % 100 == 0 or idx == total-1:
        print(f"[{idx+1}/{total}] generated")

# 4) Save full CSV
df = pd.DataFrame(results)
df.to_csv("generated_sql_queries_spider_full.csv", index=False)
print("✅ Done! Saved to generated_sql_queries_spider_full.csv")

# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import pandas as pd

# # 1) Load & take 10
# dataset = load_dataset("xlangai/spider", split="train")
# small   = dataset.select(range(10))

# # 2) Tokenizer & model
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
# model     = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)

# def generate_sql(prompt: str) -> str:
#     inputs  = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_length=1000, temperature=0.2, do_sample=True)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# # 3) Loop & include a column for the NL question
# results = []
# for idx, ex in enumerate(small):
#     instance_id = ex.get("query_id", idx)
#     nl_query    = ex["question"]
#     db_id       = ex["db_id"]

#     prompt = (
#         f"Generate an SQL query using the database name {db_id} "
#         f"for the following question:\n\n{nl_query}"
#     )

#     sql = generate_sql(prompt)
#     results.append({
#         "instance_id":   instance_id,
#         "nl_query":      nl_query,         # <<— new column
#         "generated_sql": sql
#     })
#     print(f"[{idx+1}/10] done")

# # 4) Save
# df = pd.DataFrame(results)
# df.to_csv("generated_sql_queries_spider_10.csv", index=False)
# print("✅ Saved to generated_sql_queries_spider_10.csv")