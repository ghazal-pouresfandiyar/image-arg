# Image-Arg

The annotation UI in `src/annotate_UI.py` uses only the Python standard library, so no `requirements.txt` is needed for this project.


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