from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# 1) Load CoSQL train split
dataset = load_dataset("karlen532/cosql", split="train")

# 2) Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True
)
print("model loaded")

MAX_TOTAL_TOKENS = 2000
MAX_NEW_TOKENS   = 500     # how many tokens you actually want to generate

def generate_sql(prompt: str) -> str:
    inputs  = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.2,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# 3) Loop & extract NL from `instruction`
results = []
skipped = 0
for idx, ex in enumerate(dataset):
    nl_query    = ex["instruction"]
    prompt      = (
        f"Generate an SQL query using the database schema in the context:\n\n"
        f"{nl_query}"
    )

    # --- Skip if this prompt + generation would exceed 2000 tokens ---
    tokenized = tokenizer(prompt, return_tensors="pt")
    total_tokens = tokenized.input_ids.size(1) + MAX_NEW_TOKENS
    if total_tokens > MAX_TOTAL_TOKENS:
        skipped += 1
        print(f"[{idx+1}/{len(dataset)}] 🔶 Skipping (needs {total_tokens} tokens)")
        continue

    sql = generate_sql(prompt)
    results.append({
        "instance_id":   idx,
        "nl_query":      nl_query,
        "generated_sql": sql
    })

    if (idx + 1 - skipped) % 100 == 0:
        print(f"[processed {idx+1 - skipped} queries, {skipped} skipped]")

# 4) Save
df = pd.DataFrame(results)
df.to_csv("generated_sql_queries_cosql_full.csv", index=False)
print(f"✅ Done! {len(results)} generated, {skipped} skipped. Saved to generated_sql_queries_cosql_full.csv")