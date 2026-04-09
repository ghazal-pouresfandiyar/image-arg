# Image-Arg

This app helps you review and edit image annotations one image at a time.

## What you will see

For each image, the right panel shows the annotation fields that you can edit.

Some fields are read-only and shown as plain text:

- `animals`
- `consequences`
- `climateaction`
- `type`
- `setting`

The text below those fields is also shown in the sidebar so you can quickly check the current values.

## How to run

From the project root, start the app with:

```bash
python3 src/annotate_UI.py
```

A browser window will open automatically.

## How to edit

The app lets you edit these fields:

- `premises`
- `facts`
- `conclusions`
- `notes`

`premises`, `facts`, and `conclusions` are lists of separate sentences.

- Use **Add sentence** to add a new sentence.
- Use **Delete** to remove a sentence.
- Press **Enter** in a sentence box to add another sentence.

`notes` is a normal text field.

## Saving changes

Use the **Save**, **Previous**, and **Next** buttons to move through the images and store your changes.

When you save:

- the CSV file is updated in `dataset/annotated.csv`
- `facts` is also written to `dataset/facts.json` for each image
- trailing spaces and extra line breaks are removed before text is stored

The app keeps a backup copy of the CSV the first time you save.

## Files used by the app

- `dataset/annotated.csv`
- `dataset/facts.json`
- `dataset/images_for_annotation/`
