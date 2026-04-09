#!/usr/bin/env python3
"""Simple local annotation UI for image-based CSV review.

Run this file with Python, then open the local address shown in the terminal.
The UI shows one image at a time, with editable annotation fields on the right.
Changes are written back to the CSV file on save.
"""

from __future__ import annotations

import csv
import html
import json
import os
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, quote, unquote, urlparse


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
CSV_PATH = DATASET_DIR / "annotated.csv"
IMAGES_DIR = DATASET_DIR / "images_for_annotation"
FACTS_JSON_PATH = DATASET_DIR / "facts.json"
BACKUP_PATH = CSV_PATH.parent / (CSV_PATH.name + ".bak")
IMAGE_EXT = ".jpg"
HOST = "127.0.0.1"
PORT = 8000

READ_ONLY_FIELDS = {"id", "hash_id", "url", "source_file", "animals", "consequences", "climateaction", "type", "setting", ""}
LIST_FIELDS = {"premises", "conclusions", "facts"}
TEXTAREA_FIELDS = {"notes"}
METADATA_FIELDS = ["animals", "consequences", "climateaction", "type", "setting"]
DISPLAY_ORDER = [
		"premises",
		"facts",
		"conclusions",
		"notes",
]


def read_csv_with_fallback(path: Path):
		encodings = ["utf-8-sig", "cp1252", "latin-1"]
		last_error = None

		for encoding in encodings:
				try:
						with path.open("r", newline="", encoding=encoding) as handle:
								reader = csv.DictReader(handle)
								if not reader.fieldnames:
										raise ValueError(f"CSV has no header: {path}")
								rows = list(reader)
								return reader.fieldnames, rows, encoding
				except UnicodeDecodeError as error:
						last_error = error

		raise UnicodeDecodeError(
				last_error.encoding,
				last_error.object,
				last_error.start,
				last_error.end,
				"Could not decode CSV with utf-8-sig, cp1252, or latin-1",
		)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]):
		if path.exists() and not BACKUP_PATH.exists():
				shutil.copy2(path, BACKUP_PATH)

		with path.open("w", newline="", encoding="utf-8") as handle:
				writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
				writer.writeheader()
				writer.writerows(rows)


def build_image_lookup(images_dir: Path):
		lookup = {}
		for name in os.listdir(images_dir):
				file_path = images_dir / name
				if file_path.is_file() and name.lower().endswith(IMAGE_EXT):
						lookup[file_path.stem] = file_path
		return lookup


def normalize_value(value):
		if value is None:
				return ""
		return str(value)


def parse_list_field(raw_value):
		text = normalize_value(raw_value).strip()
		if not text:
				return []

		try:
				parsed = json.loads(text)
		except json.JSONDecodeError:
				return [text]

		if isinstance(parsed, list):
				items = []
				for item in parsed:
						cleaned = normalize_value(item).strip()
						if cleaned:
								items.append(cleaned)
				return items

		cleaned = normalize_value(parsed).strip()
		return [cleaned] if cleaned else []


def serialize_list_field(items):
		cleaned = []
		for item in items:
				value = normalize_value(item).strip()
				if value:
						cleaned.append(value)
		return json.dumps(cleaned, ensure_ascii=False)


def read_facts_json(path: Path):
		if not path.exists():
				return {"global": [], "images": {}}

		with path.open("r", encoding="utf-8") as handle:
				data = json.load(handle)

		if isinstance(data, list):
				return {"global": [normalize_value(item).strip() for item in data if normalize_value(item).strip()], "images": {}}

		if not isinstance(data, dict):
				return {"global": [], "images": {}}

		global_facts = data.get("global", [])
		if not isinstance(global_facts, list):
				global_facts = []
		images = data.get("images", {})
		if not isinstance(images, dict):
				images = {}

		normalized_images = {}
		for image_id, items in images.items():
				if isinstance(items, list):
					normalized_images[str(image_id)] = [normalize_value(item).strip() for item in items if normalize_value(item).strip()]
		return {
				"global": [normalize_value(item).strip() for item in global_facts if normalize_value(item).strip()],
				"images": normalized_images,
		}


def write_facts_json(path: Path, data):
		with path.open("w", encoding="utf-8") as handle:
				json.dump(data, handle, ensure_ascii=False, indent=2)
				handle.write("\n")


class AnnotationStore:
		def __init__(self, csv_path: Path, images_dir: Path):
				self.csv_path = csv_path
				self.images_dir = images_dir
				self.fieldnames, self.rows, self.encoding = read_csv_with_fallback(csv_path)
				self.image_lookup = build_image_lookup(images_dir)
				self.facts_data = read_facts_json(FACTS_JSON_PATH)
				self.lock = threading.Lock()
				self.rows_with_images = [row for row in self.rows if self.has_image(row)]
				self.editable_fields = [
						field for field in self.fieldnames if field not in READ_ONLY_FIELDS
				]
				self.list_fields = [field for field in self.editable_fields if field in LIST_FIELDS]

		def has_image(self, row):
				row_id = normalize_value(row.get("id")).strip()
				return row_id in self.image_lookup

		def visible_rows(self, empty_premises_only=False):
				rows = self.rows_with_images or self.rows
				if not empty_premises_only:
						return rows
				return [row for row in rows if not parse_list_field(row.get("premises"))]

		def row_count(self, empty_premises_only=False):
				return len(self.visible_rows(empty_premises_only=empty_premises_only))

		def get_row(self, index, empty_premises_only=False):
				rows = self.visible_rows(empty_premises_only=empty_premises_only)
				if not rows:
						return None
				index = max(0, min(index, len(rows) - 1))
				return rows[index]

		def get_image_path(self, row):
				row_id = normalize_value(row.get("id")).strip()
				return self.image_lookup.get(row_id)

		def get_facts_value(self, row):
				row_id = normalize_value(row.get("id")).strip()
				facts_items = self.facts_data.get("images", {}).get(row_id, [])
				return serialize_list_field(facts_items)

		def sync_facts_for_row(self, row, facts_items):
				row_id = normalize_value(row.get("id")).strip()
				self.facts_data.setdefault("global", [])
				self.facts_data.setdefault("images", {})
				self.facts_data["images"][row_id] = facts_items

		def update_row(self, index, form_data, empty_premises_only=False):
				rows = self.visible_rows(empty_premises_only=empty_premises_only)
				if not rows:
						return None

				index = max(0, min(index, len(rows) - 1))
				target_row = rows[index]
				facts_items = None

				if "facts" in form_data:
						try:
								parsed_items = json.loads(form_data["facts"])
						except json.JSONDecodeError:
								parsed_items = [form_data["facts"]]
						if not isinstance(parsed_items, list):
								parsed_items = [parsed_items]
						facts_items = [normalize_value(item).strip() for item in parsed_items if normalize_value(item).strip()]

				for field in self.editable_fields:
						if field in form_data:
								if field in self.list_fields:
									try:
										parsed_items = json.loads(form_data[field])
									except json.JSONDecodeError:
										parsed_items = [form_data[field]]
									if not isinstance(parsed_items, list):
										parsed_items = [parsed_items]
									target_row[field] = serialize_list_field(parsed_items)
								else:
									target_row[field] = form_data[field].strip() if field in TEXTAREA_FIELDS else form_data[field]

				with self.lock:
						if facts_items is not None:
								self.sync_facts_for_row(target_row, facts_items)
								write_facts_json(FACTS_JSON_PATH, self.facts_data)
						write_csv(self.csv_path, self.fieldnames, self.rows)

				return index


STORE = AnnotationStore(CSV_PATH, IMAGES_DIR)


def render_input(field, value):
		field_name = html.escape(field or "Unnamed column")
		safe_value = html.escape(normalize_value(value))

		if field in LIST_FIELDS:
				items = parse_list_field(value)
				if not items:
						items = [""]
				note_html = ""
				if field == "facts":
						note_html = '<div class="facts-note">If you have already added a fact to knowledge, you don’t have to add it again; you can use it.</div>'

				rows_html = []
				for item in items:
						rows_html.append(
							f"""
							<div class="sentence-row" data-item-row>
								<input type="text" class="sentence-input" value="{html.escape(item)}" placeholder="Add a sentence">
								<button type="button" class="delete-item" data-remove-item aria-label="Delete sentence">Delete</button>
							</div>
							"""
						)

				return f"""
					<label class="field list-field" data-list-field="{html.escape(field)}">
						<span>{field_name}</span>
						{note_html}
						<div class="list-editor">
							<div class="sentence-list" data-list-items>
								{''.join(rows_html)}
							</div>
							<div class="buttons list-buttons">
								<button type="button" class="secondary" data-add-item>Add sentence</button>
							</div>
						</div>
						<input type="hidden" name="{html.escape(field)}" value="{html.escape(serialize_list_field(items))}" data-list-value>
					</label>
				"""

		if field in TEXTAREA_FIELDS:
				return f"""
						<label class=\"field\"> 
							<span>{field_name}</span>
							<textarea name=\"{html.escape(field)}\" rows=\"6\">{safe_value}</textarea>
						</label>
				"""

		if field in READ_ONLY_FIELDS:
				return f"""
						<label class=\"field\"> 
							<span>{field_name}</span>
							<input type=\"text\" name=\"{html.escape(field)}\" value=\"{safe_value}\" readonly>
						</label>
				"""

		return f"""
				<label class=\"field\"> 
					<span>{field_name}</span>
					<input type=\"text\" name=\"{html.escape(field)}\" value=\"{safe_value}\">
				</label>
		"""


def render_page(index, message="", empty_premises_only=False):
		row = STORE.get_row(index, empty_premises_only=empty_premises_only)
		total = STORE.row_count(empty_premises_only=empty_premises_only)

		if row is None:
				return """
				<!doctype html>
				<html>
					<head>
						<meta charset=\"utf-8\">
						<title>Annotation UI</title>
					</head>
					<body>
						<p>No annotation rows were found.</p>
					</body>
				</html>
				"""

		image_path = STORE.get_image_path(row)
		current_id = normalize_value(row.get("id"))
		position = index + 1
		url_value = normalize_value(row.get("url")).strip()
		message_html = f'<div class="message">{html.escape(message)}</div>' if message else ""
		filter_checked = "checked" if empty_premises_only else ""
		metadata_lines = [f"id : {current_id}"] + [
				f"{field} : {normalize_value(row.get(field))}"
				for field in METADATA_FIELDS
		]
		metadata_html = "<p class='record-meta'>" + "<br>".join(html.escape(line) for line in metadata_lines) + "</p>"
		if url_value:
				image_url_html = (
						'<p class="image-link-note">if the image is not clear use this link: '
						f'<a href="{html.escape(url_value)}" target="_blank" rel="noopener noreferrer">{html.escape(url_value)}</a>'
						"</p>"
				)
		else:
				image_url_html = '<p class="image-link-note">if the image is not clear use this link: -</p>'

		fields_html = []
		for field in DISPLAY_ORDER:
				if field == "facts":
						fields_html.append(render_input(field, STORE.get_facts_value(row)))
				else:
						fields_html.append(render_input(field, row.get(field)))

		for field in STORE.editable_fields:
				if field not in DISPLAY_ORDER:
						fields_html.append(render_input(field, row.get(field)))

		if image_path:
				image_html = f'<img src="/image/{quote(current_id)}" alt="Image {html.escape(current_id)}">'
		else:
				image_html = '<div class="missing">Image not found for this id.</div>'

		prev_disabled = "disabled" if position <= 1 else ""
		next_disabled = "disabled" if position >= total else ""

		return f"""
		<!doctype html>
		<html>
			<head>
				<meta charset=\"utf-8\">
				<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
				<title>Annotation UI</title>
				<style>
					:root {{
						color-scheme: light;
						--bg: #f3f4f6;
						--panel: #ffffff;
						--border: #d6dbe3;
						--text: #17212b;
						--muted: #5b6573;
						--accent: #2457d6;
					}}
					* {{ box-sizing: border-box; }}
					body {{
						margin: 0;
						font-family: Arial, Helvetica, sans-serif;
						background: var(--bg);
						color: var(--text);
					}}
					.topbar {{
						display: flex;
						gap: 16px;
						align-items: center;
						justify-content: space-between;
						padding: 14px 18px;
						border-bottom: 1px solid var(--border);
						background: #eef2f7;
					}}
					.topbar strong {{ font-size: 15px; }}
					.topbar-right {{ display: flex; align-items: center; gap: 10px; }}
					.topbar form {{ margin: 0; }}
					.jump-form {{ display: flex; align-items: center; gap: 6px; }}
					.jump-form input {{
						width: 84px;
						border: 1px solid var(--border);
						border-radius: 8px;
						padding: 6px 8px;
						font: inherit;
					}}
					.jump-form button {{
						border: 0;
						border-radius: 8px;
						padding: 6px 10px;
						font: inherit;
						cursor: pointer;
						background: #334155;
						color: #fff;
					}}
					.filter-form {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); }}
					.filter-form input {{ margin: 0; }}
					.wrap {{
						display: flex;
						min-height: calc(100vh - 58px);
					}}
					.viewer {{
						flex: 1 1 auto;
						padding: 20px;
						display: flex;
						flex-direction: column;
						gap: 12px;
						align-items: center;
						justify-content: flex-start;
					}}
					.canvas {{
						width: 100%;
						max-width: 1100px;
						min-height: 70vh;
						display: flex;
						align-items: center;
						justify-content: center;
						border: 1px solid var(--border);
						border-radius: 14px;
						background: #fff;
						overflow: hidden;
						box-shadow: 0 8px 24px rgba(17, 24, 39, 0.08);
					}}
					.canvas img {{
						display: block;
						max-width: 100%;
						max-height: 78vh;
						object-fit: contain;
						background: #fff;
					}}
					.image-link-note {{
						width: 100%;
						max-width: 1100px;
						margin: 0;
						font-size: 13px;
						line-height: 1.4;
						color: var(--muted);
						word-break: break-word;
					}}
					.facts-note {{
						font-size: 12px;
						color: var(--muted);
						line-height: 1.35;
					}}
					.image-link-note a {{
						color: var(--accent);
						text-decoration: underline;
					}}
					.sidebar {{
						width: 420px;
						background: var(--panel);
						border-left: 1px solid var(--border);
						padding: 12px 14px 14px;
						overflow-y: auto;
					}}
					.record-meta {{
						margin: 0 0 10px 0;
						padding: 8px 10px;
						border: 1px solid var(--border);
						border-radius: 10px;
						background: #f8fafc;
						color: var(--text);
						font-size: 13px;
						line-height: 1.35;
						white-space: normal;
					}}
					.sidebar form {{ display: flex; flex-direction: column; gap: 8px; }}
					.field {{ display: flex; flex-direction: column; gap: 6px; }}
					.field span {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
					.field input, .field textarea {{
						width: 100%;
						border: 1px solid var(--border);
						border-radius: 10px;
						padding: 10px 12px;
						font: inherit;
						color: var(--text);
						background: #fff;
					}}
					.list-editor {{ display: flex; flex-direction: column; gap: 10px; }}
					.sentence-list {{ display: flex; flex-direction: column; gap: 8px; }}
					.sentence-row {{ display: flex; gap: 8px; align-items: center; }}
					.sentence-row .sentence-input {{ flex: 1 1 auto; }}
					.delete-item {{
						border: 1px solid var(--border);
						border-radius: 10px;
						padding: 10px 12px;
						font: inherit;
						cursor: pointer;
						background: #fff;
						color: #7f1d1d;
					}}
					.list-buttons {{ margin-top: 0; }}
					.field textarea {{ resize: vertical; min-height: 110px; }}
					.readonly {{ background: #f9fafb; color: #4b5563; }}
					.buttons {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 6px; }}
					.buttons button {{
						border: 0;
						border-radius: 10px;
						padding: 10px 14px;
						font: inherit;
						cursor: pointer;
						background: var(--accent);
						color: #fff;
					}}
					.buttons button.secondary {{ background: #334155; }}
					.buttons button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
					.message {{
						width: 100%;
						max-width: 1100px;
						padding: 10px 12px;
						border: 1px solid #b8d7c0;
						border-radius: 10px;
						background: #ecfdf3;
						color: #166534;
						font-size: 14px;
					}}
					.missing {{
						padding: 18px;
						color: #991b1b;
						font-size: 16px;
					}}
					@media (max-width: 980px) {{
						.wrap {{ flex-direction: column; }}
						.sidebar {{ width: 100%; border-left: 0; border-top: 1px solid var(--border); }}
						.viewer {{ padding-bottom: 6px; }}
					}}
				</style>
			</head>
			<body>
				<div class="topbar">
					<strong>Image Annotation UI</strong>
					<div class="topbar-right">
						<div>Row {position} of {total}</div>
						<form class="filter-form" method="get" action="/">
							<input type="hidden" name="index" value="{index}">
							<input type="checkbox" name="empty_premises_only" value="1" {filter_checked}>
							<span>Empty premises only</span>
							<button type="submit">Apply</button>
						</form>
						<form class="jump-form" method="get" action="/">
							<input type="hidden" name="empty_premises_only" value="{1 if empty_premises_only else 0}">
							<input type="text" name="jump_row" placeholder="Row (e.g. 7)" aria-label="Jump to row">
							<button type="submit">Go</button>
						</form>
					</div>
				</div>
				<div class="wrap">
					<div class="viewer">
						{message_html}
						<div class="canvas">{image_html}</div>
						{image_url_html}
					</div>
					<aside class="sidebar">
						{metadata_html}
						<form method="post" action="/save?index={index}">
							<input type="hidden" name="index" value="{index}">
							<input type="hidden" name="empty_premises_only" value="{1 if empty_premises_only else 0}">
							{''.join(fields_html)}
							<div class="buttons">
								<button type="submit" name="action" value="prev" class="secondary" {prev_disabled}>Previous</button>
								<button type="submit" name="action" value="save">Save</button>
								<button type="submit" name="action" value="next" {next_disabled}>Next</button>
							</div>
						</form>
						<script>
						(function () {{
							function updateListField(container) {{
								const hidden = container.querySelector('[data-list-value]');
								const values = Array.from(container.querySelectorAll('.sentence-input'))
									.map((input) => input.value.trim())
									.filter((value) => value.length > 0);
								hidden.value = JSON.stringify(values);
							}}

							function createRow(value = '') {{
								const row = document.createElement('div');
								row.className = 'sentence-row';
								row.setAttribute('data-item-row', '');

								const input = document.createElement('input');
								input.type = 'text';
								input.className = 'sentence-input';
								input.placeholder = 'Add a sentence';
								input.value = value;

								const removeButton = document.createElement('button');
								removeButton.type = 'button';
								removeButton.className = 'delete-item';
								removeButton.setAttribute('data-remove-item', '');
								removeButton.setAttribute('aria-label', 'Delete sentence');
								removeButton.textContent = 'Delete';

								row.append(input, removeButton);
								return row;
							}}

							document.querySelectorAll('[data-list-field]').forEach((container) => {{
								const list = container.querySelector('[data-list-items]');
								const addButton = container.querySelector('[data-add-item]');

								addButton.addEventListener('click', () => {{
									const row = createRow('');
									list.appendChild(row);
									row.querySelector('.sentence-input').focus();
									updateListField(container);
								}});

								list.addEventListener('click', (event) => {{
									const button = event.target.closest('[data-remove-item]');
									if (!button) return;
									const row = button.closest('[data-item-row]');
									if (row) {{
										row.remove();
										if (!list.querySelector('[data-item-row]')) {{
											list.appendChild(createRow(''));
										}}
										updateListField(container);
									}}
								}});

								list.addEventListener('input', () => updateListField(container));

								list.addEventListener('keydown', (event) => {{
									if (event.key !== 'Enter') return;
									event.preventDefault();
									const row = event.target.closest('[data-item-row]');
									if (!row) return;
									const newRow = createRow('');
									row.after(newRow);
									newRow.querySelector('.sentence-input').focus();
									updateListField(container);
								}});

								updateListField(container);
							}});

							document.querySelector('form').addEventListener('submit', () => {{
								document.querySelectorAll('[data-list-field]').forEach((container) => updateListField(container));
							}});
						}})();
						</script>
					</aside>
				</div>
			</body>
		</html>
		"""


class AnnotationHandler(BaseHTTPRequestHandler):
		def log_message(self, format, *args):
				return

		def do_GET(self):
				parsed = urlparse(self.path)
				if parsed.path == "/image":
						self.send_error(404)
						return

				if parsed.path.startswith("/image/"):
						self.serve_image(parsed.path.removeprefix("/image/"))
						return

				params = parse_qs(parsed.query)
				message = params.get("message", [""])[0]
				jump_row = params.get("jump_row", [""])[0].strip()
				empty_premises_only = params.get("empty_premises_only", ["0"])[0] == "1"

				if jump_row:
						try:
								# Row numbers in the UI are 1-based for users.
								requested_row = int(jump_row)
						except ValueError:
								index = self.parse_index(params.get("index", ["0"])[0])
								message = f"Invalid row number: {jump_row}."
						else:
								total = STORE.row_count(empty_premises_only=empty_premises_only)
								if total == 0:
										index = 0
										message = "No rows available."
								elif requested_row < 1 or requested_row > total:
										index = self.parse_index(params.get("index", ["0"])[0])
										message = f"Row must be between 1 and {total}."
								else:
										index = requested_row - 1
										message = f"Jumped to row {requested_row}."
				else:
						index = self.parse_index(params.get("index", ["0"])[0])

				page = render_page(index, message, empty_premises_only=empty_premises_only)
				self.send_html(page)

		def do_POST(self):
				parsed = urlparse(self.path)
				if parsed.path != "/save":
						self.send_error(404)
						return

				length = int(self.headers.get("Content-Length", "0"))
				payload = self.rfile.read(length).decode("utf-8", errors="replace")
				form = parse_qs(payload, keep_blank_values=True)
				index = self.parse_index(form.get("index", ["0"])[0])
				action = form.get("action", ["save"])[0]
				empty_premises_only = form.get("empty_premises_only", ["0"])[0] == "1"

				row_index = STORE.update_row(
						index,
						{key: values[0] for key, values in form.items()},
						empty_premises_only=empty_premises_only,
				)
				if row_index is None:
						self.redirect("/?message=" + quote("No rows available."))
						return

				total = STORE.row_count(empty_premises_only=empty_premises_only)
				current_row = STORE.get_row(row_index, empty_premises_only=False)
				row_left_filter = empty_premises_only and current_row is not None and parse_list_field(current_row.get("premises"))
				status_message = "Saved."
				if row_left_filter:
						status_message = "Saved. This row no longer matches the empty premises filter."
				if action == "prev":
						target = max(0, row_index - 1)
						self.redirect(f"/?index={target}&empty_premises_only={1 if empty_premises_only else 0}&message=" + quote(status_message))
						return
				if action == "next":
						target = min(total - 1, row_index + 1)
						self.redirect(f"/?index={target}&empty_premises_only={1 if empty_premises_only else 0}&message=" + quote(status_message))
						return

				self.redirect(f"/?index={row_index}&empty_premises_only={1 if empty_premises_only else 0}&message=" + quote(status_message))

		def parse_index(self, raw_value):
				try:
						return int(raw_value)
				except (TypeError, ValueError):
						return 0

		def send_html(self, content):
				data = content.encode("utf-8")
				self.send_response(200)
				self.send_header("Content-Type", "text/html; charset=utf-8")
				self.send_header("Content-Length", str(len(data)))
				self.end_headers()
				self.wfile.write(data)

		def redirect(self, location):
				self.send_response(303)
				self.send_header("Location", location)
				self.end_headers()

		def serve_image(self, row_id):
				row_id = unquote(row_id)
				image_path = STORE.image_lookup.get(row_id)
				if not image_path or not image_path.exists():
						self.send_error(404, "Image not found")
						return

				data = image_path.read_bytes()
				self.send_response(200)
				self.send_header("Content-Type", "image/jpeg")
				self.send_header("Content-Length", str(len(data)))
				self.end_headers()
				self.wfile.write(data)


def main():
		if not CSV_PATH.exists():
				raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
		if not IMAGES_DIR.exists():
				raise NotADirectoryError(f"Images folder not found: {IMAGES_DIR}")

		server = ThreadingHTTPServer((HOST, PORT), AnnotationHandler)
		url = f"http://{HOST}:{PORT}/"
		print(f"Annotation UI running at {url}")
		print(f"Editing CSV: {CSV_PATH}")
		webbrowser.open(url)
		try:
				server.serve_forever()
		except KeyboardInterrupt:
				print("\nStopping annotation UI.")
		finally:
				server.server_close()


if __name__ == "__main__":
		main()
