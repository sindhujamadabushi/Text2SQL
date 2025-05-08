from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# 1) Load the full GeoQuery-tableQA train split (530 examples)
dataset = load_dataset("vaishali/geoQuery-tableQA", split="train")
print("Columns:", dataset.column_names)  # ['query','question','answer','table_names','tables','source','target']
print(f"Total examples: {len(dataset)}")

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

# 3) Loop over every example, build prompt and generate SQL
results = []
total = len(dataset)
for idx, ex in enumerate(dataset):
    instance_id = ex["query"]              # uses the dataset's 'query' field as ID
    nl_query    = ex["question"]           # the natural-language question
    prompt      = (
        f"Generate an SQL query for the following geography question:\n\n{nl_query}"
    )

    sql = generate_sql(prompt)
    results.append({
        "instance_id":   instance_id,
        "nl_query":      nl_query,
        "generated_sql": sql
    })

    # print progress every 50 examples
    if (idx + 1) % 50 == 0 or idx == total - 1:
        print(f"[{idx+1}/{total}] generated")

# 4) Save the full output
df = pd.DataFrame(results)
df.to_csv("generated_sql_queries_geo_full.csv", index=False)
print("✅ Done! Saved to generated_sql_queries_geo_full.csv")