import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

BASE = r"F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg"
input_path = f"{BASE}/dataset/featured-data/3_part_of_speech.csv"
output_path = f"{BASE}/output/extracted_arguments_ml.csv"

df = pd.read_csv(input_path, dtype=str, engine="python")

# load model (no training)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


def build_row_input(row):
    dialog = row.get("dialog", "") or ""
    narration = row.get("narration", "") or ""
    emotion = row.get("emotion", "") or ""
    sentiment = row.get("sentiment", "") or ""
    tag = row.get("tag", "") or ""

    return (
        f"dialog: {dialog}. "
        f"narration: {narration}. "
        f"emotion: {emotion}. "
        f"sentiment: {sentiment}. "
        f"tag: {tag}."
    )


def clean_output(text):
    if not text:
        return "The speaker reacts based on the situation."

    text = text.strip()

    # if model echoed prompt-like content
    if "dialog:" in text or "emotion:" in text:
        return "The speaker reacts based on the situation."

    return text


def generate_argument(prompt):
    inputs = tokenizer([prompt], return_tensors="pt",
                       truncation=True, max_length=256)

    outputs = model.generate(
        inputs.input_ids,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return clean_output(decoded)


arguments = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    prompt = build_row_input(row)
    arg = generate_argument(prompt)
    arguments.append(arg)

df["argument"] = arguments
df.to_csv(output_path, index=False)

print("Saved ML-based arguments:", output_path)
