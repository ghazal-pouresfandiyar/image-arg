import pandas as pd
from textblob import TextBlob

BASE = r"F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/dataset"

input_path = f"{BASE}/no_tags_expanded.csv"
output_path = f"{BASE}/featured-data/1_dialog_sentiment.csv"

df = pd.read_csv(input_path, dtype=str, engine="python")

# compute TextBlob polarity and subjectivity for each dialog text
df["sentiment"] = df["dialog"].fillna("").apply(
    lambda text: TextBlob(text).sentiment.polarity)
df["subjectivity"] = df["dialog"].fillna("").apply(
    lambda text: TextBlob(text).sentiment.subjectivity)

df.to_csv(output_path, index=False)

print("Saved sentiment file to:", output_path)
