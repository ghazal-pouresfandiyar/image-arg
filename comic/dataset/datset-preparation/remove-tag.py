import re
import ast
import json
import pandas as pd

BASE = r"F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/dataset"


input_path = f"{BASE}/filtered_data.csv"
output_path = f"{BASE}/no_tags_expanded.csv"


# Simple parser: if cell looks like a Python list (starts with '['), parse it;
# otherwise treat the whole cell as a single string. No try/except used.
def parse_dialog_items(cell):
    """Return list of (tag, content) tuples parsed from the dialog cell.

    - If cell looks like a Python list literal it's parsed with ast.literal_eval.
    - Each item is checked for a leading tag like <TAG content>.
    - If no tag is present, `tag` is an empty string and `content` is the item text.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []

    if isinstance(cell, (list, tuple)):
        items = list(cell)
    elif isinstance(cell, str) and cell.strip().startswith("[") and cell.strip().endswith("]"):
        items = ast.literal_eval(cell)
    else:
        items = [cell]

    out = []
    tag_re = re.compile(r"^\s*<([A-Z0-9\-_]+)\s*([^>]*)>\s*$")
    for e in items:
        s = "" if e is None else str(e)
        m = tag_re.match(s)
        if m:
            tag = m.group(1)
            content = m.group(2).strip()
        else:
            # try to remove any inline uppercase tag and keep the rest
            content = re.sub(r"<[A-Z0-9\-_]+\s*([^>]*)>", r"\1", s).strip()
            tag = ""

        # strip surrounding quotes and whitespace
        content = content.strip().strip('"').strip("'")
        out.append((tag, content))

    return out


df = pd.read_csv(input_path, dtype=str, engine="python")

# Build expanded rows: one dialog item per output row, with a `tag` column.
rows = []
for _, row in df.iterrows():
    parsed = parse_dialog_items(row.get("dialog"))
    if not parsed:
        # keep row but empty dialog and tag
        new = row.to_dict()
        new["dialog"] = ""
        new["tag"] = ""
        rows.append(new)
        continue

    for tag, content in parsed:
        new = row.to_dict()
        new["dialog"] = content
        new["tag"] = tag
        rows.append(new)

out_df = pd.DataFrame(rows)
out_df.to_csv(output_path, index=False)

print("Saved expanded file to:", output_path)
