# Image-Arg

This app helps you review and edit image annotations one image at a time.

## What you will see

For each image, the right panel shows the annotation fields you can edit. The image URL is shown under the picture in case the image preview is hard to read.

Some fields are shown as plain text and cannot be edited:

- `animals`
- `consequences`
- `climateaction`
- `type`
- `setting`

The main metadata for the current row is shown in a short text block above the editable fields.

## How to run

From the project root, start the app with:

```bash
python3 src/annotate_UI.py
```

A browser window will open automatically.

## Moving around

Use the row jump box at the top to go straight to a specific row number, such as row 7.

You can also turn on the `Empty premises only` filter to show only rows where `premises` has not been filled in yet.

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

If a fact has already been added to the knowledge file, you can leave the `facts` field as it is.

## Saving changes

Use the **Save**, **Previous**, and **Next** buttons to move through the images and store your changes.

When you save:

- the CSV file is updated in `dataset/annotated.csv`
- `facts` is written to `dataset/facts.json`
- trailing spaces and extra line breaks are removed before text is stored

The app keeps a backup copy of the CSV the first time you save.
