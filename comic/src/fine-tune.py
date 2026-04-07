from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

df = pd.read_csv(
    "F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/dataset/featured-data/3_part_of_speech.csv", dtype=str, engine="python")


def pseudo_target(row):
    try:
        sentiment = float(row.get("sentiment", 0) or 0)
    except:
        sentiment = 0

    emotion = row.get("emotion", "")

    if sentiment < 0:
        return "The speaker feels negative because of the situation."
    if sentiment > 0.2:
        return "The speaker seems positive or satisfied with the situation."
    if isinstance(emotion, str):
        if "anger" in emotion:
            return "The speaker is angry because something upset them."
        if "fear" in emotion:
            return "The speaker feels afraid due to the situation."
    return "The speaker reacts based on the context."


df["target_argument"] = df.apply(pseudo_target, axis=1)


def build_input(row):
    return (
        f"dialog: {row['dialog']} "
        f"narration: {row['narration']} "
        f"emotion: {row['emotion']} "
        f"sentiment: {row['sentiment']} "
        f"tag: {row['tag']}"
    )


df["input_text"] = df.apply(build_input, axis=1)
df["target_text"] = df["target_argument"]

train_df, test_df = train_test_split(
    df[["input_text", "target_text"]], test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

tokenizer = T5Tokenizer.from_pretrained("t5-small")


def tokenize(example):
    inputs = tokenizer(
        example["input_text"], truncation=True, padding="max_length", max_length=256)
    targets = tokenizer(
        example["target_text"], truncation=True, padding="max_length", max_length=64)
    inputs["labels"] = targets["input_ids"]
    return inputs


train_dataset = train_dataset.map(tokenize, batched=False)
test_dataset = test_dataset.map(tokenize, batched=False)

model = T5ForConditionalGeneration.from_pretrained("t5-small")

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./t5_argument_finetuned",
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=10,
    ),
    train_dataset=train_dataset
)

trainer.train()

model.save_pretrained("./t5_argument_finetuned")
tokenizer.save_pretrained("./t5_argument_finetuned")

model = T5ForConditionalGeneration.from_pretrained("./t5_argument_finetuned")
tokenizer = T5Tokenizer.from_pretrained("./t5_argument_finetuned")


def generate_argument(dialog, narration="", sentiment=0.0, emotion=""):
    input_text = f"dialog: {dialog} narration: {narration} emotion: {emotion} sentiment: {sentiment}"
    inputs = tokenizer([input_text], return_tensors="pt", truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def predict(row):
    dialog = row["input_text"]
    return generate_argument(dialog, "", 0.0, "")


test_df["predicted_argument"] = test_df.apply(predict, axis=1)

test_df.to_csv(
    "F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/output/test_predictions.csv", index=False)

print("predictions saved")
