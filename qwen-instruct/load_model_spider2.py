import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Load tokenizer and model (include trust_remote_code=True if needed)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)

print("model loaded")

def generate_sql(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=1000, temperature=0.2, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

results = []
input_file = "spider2-lite.jsonl"  # File containing your JSON lines

with open(input_file, "r") as f:
    for line in f:
        instance = json.loads(line)
        instance_id = instance.get("instance_id", "unknown")
        question = instance.get("question", "")
        db = instance.get("db", "")
        # Construct prompt: questiaon followed by a statement of the database
        prompt = F"Generate an SQL query using the database name {db} for the following question: {question}"
        # prompt = f"You are an SQL query generator. Given a natural language question, generate a corresponding SQL query. Your Natural language question is '{question}'."
        generated_answer = generate_sql(prompt)
        results.append({
            "instance_id": instance_id,
            "generated_sql": generated_answer
        })
        print(f"Generated query for instance ID {instance_id}")
        

# Save results to a CSV file
output_df = pd.DataFrame(results)
output_df.to_csv("generated_sql_queries_spider2.csv", index=False)
print("SQL generation complete. Results saved to 'generated_sql_queries.csv'.")