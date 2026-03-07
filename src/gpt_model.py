from transformers import pipeline
import pandas as pd

# Load generation model (you can change the model)
generator = pipeline("text-generation", model="gpt2")


def build_prompt(dialog, narration):
    return (
        "Generate an argument explaining the situation in the comic.\n"
        "Argument:\n"
    )


def generate_argument(dialog, narration):
    prompt = build_prompt(dialog, narration)

    result = generator(
        prompt,
        max_length=80,
        num_return_sequences=1,
        temperature=0.7
    )

    return result[0]["generated_text"].replace(prompt, "").strip()


# Load dataset
df = pd.read_csv(
    "F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/dataset/no_tags_data.csv")

# Generate arguments
df["argument"] = df.apply(lambda row: generate_argument(
    row["dialog"], row["narration"]), axis=1)

# Save results
df.to_csv("F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/output/gpt_arguments.csv", index=False)

print(df[["argument"]].head())
