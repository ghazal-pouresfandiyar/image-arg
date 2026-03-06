import re
import ast
import json
import pandas as pd

BASE = r"F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/dataset"

BASE = r"F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/dataset"


input_path = f"{BASE}/filtered_data.csv"
output_path = f"{BASE}/no_tags_data.csv"


# Simple parser: if cell looks like a Python list (starts with '['), parse it;
# otherwise treat the whole cell as a single string. No try/except used.
def remove_uppercase_tags(cell):
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []

    if isinstance(cell, (list, tuple)):
        items = list(cell)
    elif isinstance(cell, str) and cell.strip().startswith("[") and cell.strip().endswith("]"):
        # assume well-formed list literal
        items = ast.literal_eval(cell)
    else:
        items = [cell]

    out = []
    for e in items:
        s = str(e)
        # replace each tag like <INFORM some text> with 'some text'
        cleaned = re.sub(r"<[A-Z0-9\-_]+\s*([^>]*)>", r"\1", s)
        # strip surrounding quotes and whitespace
        cleaned = cleaned.strip().strip('"').strip("'")
        out.append(cleaned)

    return out


df = pd.read_csv(input_path, dtype=str, engine="python")
df["dialog"] = df["dialog"].apply(remove_uppercase_tags)
# encode lists as JSON so strings use double quotes
df["dialog"] = df["dialog"].apply(
    lambda lst: json.dumps(lst, ensure_ascii=False))
df.to_csv(output_path, index=False)

print("Saved cleaned file to:", output_path)
