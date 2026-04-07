import json
import pandas as pd
from transformers import pipeline

BASE = r"F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/dataset/featured-data"

input_path = f"{BASE}/1_dialog_sentiment.csv"
output_path = f"{BASE}/2_emotion_detection.csv"

df = pd.read_csv(input_path, dtype=str, engine="python")

emotion = pipeline("text-classification",
                   model="j-hartmann/emotion-english-distilroberta-base",
                   top_k=None)


def get_emotions(text):
    if not text or text.strip() == "":
        return "{}"
    result = emotion(text)[0]
    scores = {item["label"]: round(item["score"], 4) for item in result}
    return json.dumps(scores)


df["emotion"] = df["dialog"].fillna("").apply(get_emotions)

df.to_csv(output_path, index=False)

print("Saved emotion file to:", output_path)
