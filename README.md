# Image-Arg

The annotation UI in `src/annotate_UI.py` uses only the Python standard library, so no `requirements.txt` is needed for this project.

For multimodal feature generation in `src/CLIP_embeddings.py`, install dependencies from `requirements.txt`.

## Multimodal feature setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Generate image and metadata features:

```bash
python3 src/CLIP_embeddings.py
```

What this script does:

- Uses pretrained CLIP `openai/clip-vit-base-patch32` (ViT-B/32)
- Extracts one image embedding per row from `dataset/images_for_annotation/`
- Builds one-hot metadata features from `animals`, `consequences`, `climateaction`, `type`, and `setting`
- Saves image features to `dataset/features/clip_image_embeddings.npy`
- Saves metadata features to `dataset/features/metadata_onehot.npy`
- Saves concatenated multimodal features to `dataset/features/multimodal_features.npy`
- Saves aligned row index to `dataset/features/feature_index.csv`
- Saves feature config to `dataset/features/feature_config.json`
- Drops rows when the image file is missing or unreadable


## Run the UI

From the repository root, run:

```bash
python3 src/annotate_UI.py
```

The script starts a local web server, opens the browser automatically, and shows one image at a time with editable fields in the right panel.

## What to edit

The annotation table includes fields such as `animals`, `consequences`, `climateaction`, `type`, `setting`, `first_argument`, `second_argument`, and `more`.

Please just edit the last 3 fields.

If you think the other fields should be edited or something is wrong with the image or the argument, write a note in `more` section.

## Saving

Use the buttons in the UI to move between items and save your changes. When you save for the first time, the app creates a backup file next to the CSV.