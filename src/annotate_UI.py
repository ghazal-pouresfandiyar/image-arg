#!/usr/bin/env python3
"""Simple local annotation UI for image-based CSV review.

Run this file with Python, then open the local address shown in the terminal.
The UI shows one image at a time, with editable annotation fields on the right.
Changes are written back to the CSV file on save.
"""

from __future__ import annotations

import csv
import html
import os
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, quote, unquote, urlparse


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
CSV_PATH = DATASET_DIR / "annotated.csv"
IMAGES_DIR = DATASET_DIR / "images_for_annotation"
BACKUP_PATH = CSV_PATH.parent / (CSV_PATH.name + ".bak")
IMAGE_EXT = ".jpg"
HOST = "127.0.0.1"
PORT = 8000

READ_ONLY_FIELDS = {"id", "hash_id", "url", "source_file", ""}
TEXTAREA_FIELDS = {"first_argument", "second_argument", "more"}
DISPLAY_ORDER = [
		"id",
		"hash_id",
		"url",
		"animals",
		"consequences",
		"climateaction",
		"type",
		"setting",
		"source_file",
		"first_argument",
		"second_argument",
		"more"
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


class AnnotationStore:
		def __init__(self, csv_path: Path, images_dir: Path):
				self.csv_path = csv_path
				self.images_dir = images_dir
				self.fieldnames, self.rows, self.encoding = read_csv_with_fallback(csv_path)
				self.image_lookup = build_image_lookup(images_dir)
				self.lock = threading.Lock()
				self.rows_with_images = [row for row in self.rows if self.has_image(row)]
				self.editable_fields = [
						field for field in self.fieldnames if field not in READ_ONLY_FIELDS
				]

		def has_image(self, row):
				row_id = normalize_value(row.get("id")).strip()
				return row_id in self.image_lookup

		def visible_rows(self):
				return self.rows_with_images or self.rows

		def row_count(self):
				return len(self.visible_rows())

		def get_row(self, index):
				rows = self.visible_rows()
				if not rows:
						return None
				index = max(0, min(index, len(rows) - 1))
				return rows[index]

		def get_image_path(self, row):
				row_id = normalize_value(row.get("id")).strip()
				return self.image_lookup.get(row_id)

		def update_row(self, index, form_data):
				rows = self.visible_rows()
				if not rows:
						return None

				index = max(0, min(index, len(rows) - 1))
				target_row = rows[index]

				for field in self.editable_fields:
						if field in form_data:
								target_row[field] = form_data[field]

				with self.lock:
						write_csv(self.csv_path, self.fieldnames, self.rows)

				return index


STORE = AnnotationStore(CSV_PATH, IMAGES_DIR)


def render_input(field, value):
		field_name = html.escape(field or "Unnamed column")
		safe_value = html.escape(normalize_value(value))

		if field in TEXTAREA_FIELDS:
				return f"""
						<label class=\"field\"> 
							<span>{field_name}</span>
							<textarea name=\"{html.escape(field)}\" rows=\"6\">{safe_value}</textarea>
						</label>
				"""

		if field == "url":
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


def render_page(index, message=""):
		row = STORE.get_row(index)
		total = STORE.row_count()

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
		url_value = normalize_value(row.get("url"))
		message_html = f'<div class="message">{html.escape(message)}</div>' if message else ""

		fields_html = []
		for field in DISPLAY_ORDER:
				if field in STORE.editable_fields:
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
					.meta {{
						width: 100%;
						max-width: 1100px;
						display: flex;
						flex-wrap: wrap;
						gap: 12px;
						color: var(--muted);
						font-size: 13px;
					}}
					.sidebar {{
						width: 420px;
						background: var(--panel);
						border-left: 1px solid var(--border);
						padding: 18px;
						overflow-y: auto;
					}}
					.sidebar form {{ display: flex; flex-direction: column; gap: 12px; }}
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
					<div>Row {position} of {total} | ID {html.escape(current_id)}</div>
				</div>
				<div class="wrap">
					<div class="viewer">
						{message_html}
						<div class="canvas">{image_html}</div>
						<div class="meta">
							<div><b>Image id:</b> {html.escape(current_id)}</div>
							<div><b>Source URL:</b> {html.escape(url_value) if url_value else "-"}</div>
						</div>
					</div>
					<aside class="sidebar">
						<form method="post" action="/save?index={index}">
							<input type="hidden" name="index" value="{index}">
							{''.join(fields_html)}
							<div class="buttons">
								<button type="submit" name="action" value="prev" class="secondary" {prev_disabled}>Previous</button>
								<button type="submit" name="action" value="save">Save</button>
								<button type="submit" name="action" value="next" {next_disabled}>Next</button>
							</div>
						</form>
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
				index = self.parse_index(params.get("index", ["0"])[0])
				message = params.get("message", [""])[0]
				page = render_page(index, message)
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

				row_index = STORE.update_row(index, {key: values[0] for key, values in form.items()})
				if row_index is None:
						self.redirect("/?message=" + quote("No rows available."))
						return

				total = STORE.row_count()
				if action == "prev":
						target = max(0, row_index - 1)
						self.redirect(f"/?index={target}&message=" + quote("Saved."))
						return
				if action == "next":
						target = min(total - 1, row_index + 1)
						self.redirect(f"/?index={target}&message=" + quote("Saved."))
						return

				self.redirect(f"/?index={row_index}&message=" + quote("Saved."))

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
