import os
import sqlite3
import json
import pandas as pd
from tqdm import tqdm
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

# Define Vanna class
class UnifiedVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

# Initialize Vanna with your OpenAI key
vn = UnifiedVanna(config={
    "api_key": "",   # Replace with your actual key
    "model": "gpt-4-turbo"
})

# Directory with .sqlite files
sqlite_dir = "databases"

# Step 1: Train on all schemas from all DBs
for file in os.listdir(sqlite_dir):
    if file.endswith(".sqlite"):
        db_path = os.path.join(sqlite_dir, file)
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND type='table';")
            ddl_statements = cursor.fetchall()
            for ddl_row in ddl_statements:
                if ddl_row[0]:
                    vn.train(ddl=ddl_row[0])
            conn.close()
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

# Step 2: Load JSONL and generate SQL
spider_path = "spider2-lite.jsonl"
with open(spider_path, "r", encoding="utf-8") as f:
    examples = [json.loads(line) for line in f if line.strip()]

output_rows = []
for ex in tqdm(examples):
    question = ex.get("question", "")
    instance_id = ex.get("instance_id", "")
    try:
        sql = vn.generate_sql(question)
        output_rows.append({
            "instance_id": instance_id,
            "question": question,
            "vanna_sql": sql
        })
    except Exception as e:
        output_rows.append({
            "instance_id": instance_id,
            "question": question,
            "vanna_sql": f"Error: {str(e)}"
        })

# Step 3: Save to output/vanna_generated_queries.csv
os.makedirs("output", exist_ok=True)
pd.DataFrame(output_rows).to_csv("output/vanna_generated_queries.csv", index=False)

print("✅ Saved to output/vanna_generated_queries.csv")



