# Climate Annotation UI

Run the annotation tool from this folder with:

```bash
python3 src/annotate_UI.py
```

The app opens a local browser page. It shows one image at a time, with editable annotation fields in the right sidebar. Saving writes changes back to `src/dataset/annotated.csv`.

# Visual Annotation Tool

This folder contains a simple local annotation interface for reviewing climate images by id and editing annotation text.

## File

- `annotate_UI.py`

## What It Does

- Loads rows from `dataset/annotated.csv`
- Matches each row by `id` to images in `dataset/images_for_annotation/` (for example `0.jpg`, `1.jpg`)
- Shows one image at a time
- Displays editable fields in a white right sidebar
- Supports `Previous`, `Save`, and `Next`
- Saves edits back to the same CSV file
- Creates a backup file on first save: `annotated.csv.bak`

## Run

From the `climate` directory:

```bash
python3 src/annotate_UI.py
```

Then open the local URL printed in the terminal (usually `http://127.0.0.1:8000/`).

## Notes

- The UI only uses Python standard library modules.
- Stop the server with `Ctrl+C` in the terminal.
