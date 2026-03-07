import ast
import pandas as pd
import os


def parse_list(cell):
    """
    Convert string list representation to Python list.
    If already list, return as-is.
    """
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        try:
            return ast.literal_eval(cell)
        except:
            return [cell]
    return []


def generate_argument(dialog, narration):
    """
    Simple rule-based argument generator.
    Uses first dialog and narration as evidence.
    """
    dialog_text = parse_list(dialog)
    narration_text = parse_list(narration)

    dialog_str = dialog_text[0] if dialog_text else "no dialogue"
    narration_str = narration_text[0] if narration_text else "no narration"

    # Basic heuristic for emotion (very simple)
    if any(word in dialog_str.lower() for word in ["!", "angry", "no", "can't", "never"]):
        emotion = "angry or conflicted"
    elif any(word in dialog_str.lower() for word in ["love", "happy", "great", "yes"]):
        emotion = "positive or happy"
    else:
        emotion = "neutral"

    return (
        f"The character seems {emotion} because the dialogue says '{dialog_str}' "
        f"and the narration indicates '{narration_str}'."
    )


# Example usage with dataframe
df = pd.read_csv(
    "F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/dataset/no_tags_data.csv")

df["argument"] = df.apply(lambda row: generate_argument(
    row["dialog"], row["narration"]), axis=1)
os.makedirs(
    "F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/output", exist_ok=True)
df.to_csv("F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/output/rule_based_arguments.csv", index=False)
