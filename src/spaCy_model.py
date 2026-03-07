import spacy
import ast
import pandas as pd
import os

# Load spaCy model (make sure you installed it: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")


def parse_list(cell):
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        try:
            return ast.literal_eval(cell)
        except:
            return [cell]
    return []


def detect_emotion(text):
    """
    Very simple emotion heuristic using keywords.
    You can extend this with ML or sentiment models.
    """
    text = text.lower()

    if any(word in text for word in ["lie", "betray", "can't believe", "no", "hate"]):
        return "angry or betrayed"
    if any(word in text for word in ["love", "happy", "great", "yes"]):
        return "happy"
    if any(word in text for word in ["sorry", "apologize", "regret"]):
        return "remorseful"
    return "neutral"


def generate_argument(dialog, narration):
    dialog_text = parse_list(dialog)
    narration_text = parse_list(narration)

    dialog_str = dialog_text[0] if dialog_text else ""
    narration_str = narration_text[0] if narration_text else ""

    # Combine for evidence
    evidence = f"{dialog_str} {narration_str}".strip()

    doc = nlp(evidence)
    emotion = detect_emotion(evidence)

    # Use spaCy to extract meaningful tokens (optional)
    keywords = [
        token.text for token in doc if token.is_alpha and not token.is_stop]
    keyword_str = ", ".join(keywords[:5]) if keywords else "the situation"

    return (
        f'The character is {emotion} because the dialogue and narration mention '
        f'"{dialog_str}" and suggest {keyword_str}.'
    )


# Example usage
df = pd.read_csv(
    "F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/dataset/no_tags_data.csv")

df["argument"] = df.apply(lambda row: generate_argument(
    row["dialog"], row["narration"]), axis=1)

os.makedirs(
    "F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/output", exist_ok=True)
# Save results
df.to_csv("F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/output/spaCyarguments.csv",
          index=False)

print(df[["dialog", "narration", "argument"]].head())
