import pandas as pd
from textblob import TextBlob

BASE = r"F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/dataset"

input_path = f"{BASE}/no_tags_expanded.csv"
output_path = f"{BASE}/featured-data/semantic_polarity.csv"

df = pd.read_csv(input_path, dtype=str, engine="python")

# compute TextBlob polarity and subjectivity for each dialog text
df["polarity"] = df["dialog"].fillna("").apply(
    lambda text: TextBlob(text).sentiment.polarity)

# Subjectivity is a float within the range [0.0, 1.0] where:
# 0.0	completely objective (fact)
# 1.0	completely subjective (opinion / emotion)
df["subjectivity"] = df["dialog"].fillna("").apply(
    lambda text: TextBlob(text).sentiment.subjectivity)

df.to_csv(output_path, index=False)

print("Saved polarity file to:", output_path)
