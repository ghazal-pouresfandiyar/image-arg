import re
import json
import pandas as pd
import spacy

BASE = r"F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg/dataset/featured-data"

input_path = f"{BASE}/2_emotion_detection.csv"
output_path = f"{BASE}/3_part_of_speech.csv"

df = pd.read_csv(input_path, dtype=str, engine="python")

nlp = spacy.load("en_core_web_lg")


def get_addressee(text, tag):
    if not text or text.strip() == "":
        return ""
    match = re.match(r'^([a-z]+(?:\s[a-z]+)*),', text, re.IGNORECASE)
    if match:
        return match.group(1)
    if tag == "ORDER":
        doc = nlp(text)
        for token in doc:
            if token.dep_ == "vocative" or (token.dep_ == "npadvmod" and token.pos_ == "PROPN"):
                return token.text
    return ""


def strip_addressee(text, addressee):
    if addressee and text.startswith(addressee + ","):
        return text[len(addressee) + 1:].strip()
    return text


def get_pos(text):
    if not text or text.strip() == "":
        return "[]"
    doc = nlp(text)
    tokens = [{"text": t.text, "pos": t.pos_,
               "dep": t.dep_, "head": t.head.text} for t in doc]
    return json.dumps(tokens)


def get_svo(text, addressee):
    if not text or text.strip() == "":
        return ""
    clean = strip_addressee(text, addressee)
    doc = nlp(clean)
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            verb = token.text
            subj = ""
            obj = ""
            voice = "active"
            for child in token.children:
                if child.dep_ == "nsubjpass":
                    subj = child.text
                    voice = "passive"
                elif child.dep_ == "nsubj":
                    subj = child.text
                if child.dep_ in ["dobj", "pobj", "ccomp", "xcomp"]:
                    obj = " ".join([t.text for t in child.subtree])
            if subj.lower() == "you" and addressee:
                subj = addressee
            if subj or obj:
                return f"{subj} + {verb}({voice}) + {obj}"
    return ""


df["addressee"] = df.apply(lambda r: get_addressee(
    r.get("dialog", ""), r.get("tag", "")), axis=1)
df["pos"] = df["dialog"].fillna("").apply(get_pos)
df["svo"] = df.apply(lambda r: get_svo(
    r.get("dialog", ""), r.get("addressee", "")), axis=1)

df.to_csv(output_path, index=False)

print("Saved POS file to:", output_path)
