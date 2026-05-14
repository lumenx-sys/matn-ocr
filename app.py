import os
import base64
import time
import tempfile
import threading
import io

from flask import Flask, request, jsonify, send_file
import anthropic
import arabic_reshaper
from bidi.algorithm import get_display
from pdf2image import convert_from_path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

app = Flask(__name__)

# ── Font registration ─────────────────────────────────────────────────────────
def register_font():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont("Arabic", path))
            return
    import urllib.request
    font_path = "/tmp/arabic_font.ttf"
    if not os.path.exists(font_path):
        url = "https://cdn.jsdelivr.net/npm/@fontsource/amiri@5.0.8/files/amiri-arabic-400-normal.ttf"
        urllib.request.urlretrieve(url, font_path)
    pdfmetrics.registerFont(TTFont("Arabic", font_path))

register_font()

# ── Prompts ───────────────────────────────────────────────────────────────────
TRANSCRIBE_PROMPT = """You are a precise Arabic OCR transcription engine.
Your only job is to transcribe exactly what you see in the image, character by character.
Rules:
- Transcribe the Arabic text EXACTLY as it appears
- Preserve all diacritics (harakat) exactly as they appear
- Preserve line breaks and punctuation as they appear
- If a word or character is unclear or damaged, write [?] in its place
- Do not add commentary, explanations, or notes
- Do not translate anything
- Output only the transcribed text, nothing else"""

DETECT_PROMPT = """You are an expert in Arabic literature and Islamic sciences.
Look at this Arabic text and identify the book's context in 1-2 sentences.
Describe: the genre, subject matter, scholarly tradition, time period, and any relevant terminology conventions a translator should know.
Be specific and concise. Output only the description, nothing else."""


def build_translation_prompt(target_language, context):
    return f"""You are a professional Arabic-to-{target_language} translator.
The text you are translating is: {context}
Translate the given Arabic text into {target_language} as LITERALLY as possible.
Rules:
- Translate LITERALLY — stay as close to the original Arabic wording and structure as possible
- Do NOT paraphrase, summarize, simplify, or interpret
- Do NOT add explanations, footnotes, or commentary
- Use translation conventions appropriate for the type of text described above
- Preserve technical terminology in Arabic transliterated form (e.g. wudu, salah, fiqh, qadi, imam, matn, sawm, zakat)
- Preserve Arabic proper nouns, titles, and honorifics as-is (Ibn, Abu, Abd, Sheikh)
- Use ONLY plain ASCII characters — no curly quotes, diacritical marks, or special punctuation
- Preserve sentence structure and paragraph breaks as in the original
- Output only the translation, nothing else"""


# ── Core helpers ──────────────────────────────────────────────────────────────
def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def call_claude(client, **kwargs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
            else:
                raise e


def transcribe_page(client, img):
    img_b64 = image_to_base64(img)
    msg = call_claude(client,
        model="claude-sonnet-4-6", max_tokens=4096,
        system=TRANSCRIBE_PROMPT,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/jpeg", "data": img_b64}},
            {"type": "text", "text": "Please transcribe all Arabic text on this page exactly as it appears."}
        ]}]
    )
    return msg.content[0].text


def translate_text(client, arabic_text, translation_prompt):
    msg = call_claude(client,
        model="claude-sonnet-4-6", max_tokens=4096,
        system=translation_prompt,
        messages=[{"role": "user", "content": arabic_text}]
    )
    return msg.content[0].text


def fix_arabic(text):
    return get_display(arabic_reshaper.reshape(text))


def sanitize(text):
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2014": "-", "\u2013": "-",
        "\u2026": "...",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("ascii", errors="ignore").decode("ascii")


_style_counter = [0]

def get_styles():
    """Create styles with unique names to avoid ReportLab duplicate registration errors."""
    _style_counter[0] += 1
    n = _style_counter[0]
    base = getSampleStyleSheet()
    heading = ParagraphStyle(f"MatnHeading{n}", parent=base["Heading2"], spaceAfter=6)
    arabic  = ParagraphStyle(f"MatnArabic{n}", parent=base["Normal"],
                              fontName="Arabic", fontSize=12, leading=22,
                              spaceAfter=4, alignment=2)
    trans   = ParagraphStyle(f"MatnTrans{n}",  parent=base["Normal"],
                              fontSize=11, leading=18, spaceAfter=4, alignment=0)
    return heading, arabic, trans


def add_arabic_page(story, page_num, text, styles):
    heading, arabic_style, _ = styles
    story.append(Paragraph(f"— Page {page_num} (Arabic) —", heading))
    story.append(Spacer(1, 4))
    for line in text.split("\n"):
        line = line.strip()
        if line:
            line = fix_arabic(line)
            line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(line, arabic_style))
        else:
            story.append(Spacer(1, 6))
    story.append(Spacer(1, 12))


def add_translation_page(story, page_num, text, lang, styles):
    heading, _, trans_style = styles
    story.append(Paragraph(f"— Page {page_num} ({lang.title()} Translation) —", heading))
    story.append(Spacer(1, 4))
    for line in text.split("\n"):
        line = line.strip()
        if line:
            line = sanitize(line)
            line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(line, trans_style))
        else:
            story.append(Spacer(1, 6))
    story.append(Spacer(1, 12))


def build_pdf_bytes(story):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    doc.build(story)
    return buf.getvalue()


# ── Job storage (in-memory, fine for single-server Railway deploy) ─────────────
jobs = {}  # job_id -> {"status": ..., "log": [...], "outputs": {...}, "error": ...}


def run_job(job_id, api_key, pdf_bytes, start_page, end_page,
            do_translate, target_lang, mode, manual_context, request_dpi):
    job = jobs[job_id]

    def log(msg):
        job["log"].append(msg)

    tmp_pdf = None
    try:
        client = anthropic.Anthropic(api_key=api_key)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_pdf = f.name

        dpi = int(request_dpi) if request_dpi and request_dpi.isdigit() else 200
        log(f"📥 PDF received — starting conversion...")
        log(f"🔄 Converting PDF pages to images (quality: {'high' if dpi == 300 else 'fast'})...")
        log(f"   This may take 1-3 minutes for a full book. Please wait...")
        conv_kwargs = dict(dpi=dpi)
        if start_page: conv_kwargs["first_page"] = start_page
        if end_page:   conv_kwargs["last_page"]  = end_page
        # Convert in a sub-thread with a 3-minute timeout
        images_result = [None]
        convert_error = [None]

        def _do_convert():
            try:
                images_result[0] = convert_from_path(tmp_pdf, **conv_kwargs)
            except Exception as e:
                convert_error[0] = e

        convert_thread = threading.Thread(target=_do_convert, daemon=True)
        convert_thread.start()
        convert_thread.join(timeout=180)  # wait up to 3 minutes

        if convert_thread.is_alive():
            raise TimeoutError("PDF conversion timed out after 3 minutes. Try using a page range (e.g. pages 1-10) instead of the full book.")
        if convert_error[0]:
            raise convert_error[0]

        images = images_result[0]

        base_page = start_page if start_page else 1
        page_numbers = list(range(base_page, base_page + len(images)))
        log(f"✅ {len(images)} pages ready")

        translation_prompt = None
        if do_translate:
            if manual_context and manual_context.strip():
                context = manual_context.strip()
                log(f"📖 Using manual context: {context}")
            else:
                log("🔍 Auto-detecting book context...")
                img_b64 = image_to_base64(images[0])
                msg = call_claude(client,
                    model="claude-sonnet-4-6", max_tokens=300,
                    system=DETECT_PROMPT,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64",
                         "media_type": "image/jpeg", "data": img_b64}},
                        {"type": "text", "text": "What is the context of this text?"}
                    ]}]
                )
                context = msg.content[0].text.strip()
                log(f"  📖 Detected: {context}")
            translation_prompt = build_translation_prompt(target_lang, context)

        pages_data = []
        for i, (img, pnum) in enumerate(zip(images, page_numbers)):
            log(f"🤖 Transcribing page {pnum} ({i+1}/{len(images)})...")

            # Run transcription in a sub-thread with 2-minute timeout
            transcribe_result = [None]
            transcribe_error = [None]
            def _do_transcribe(img=img):
                try:
                    transcribe_result[0] = transcribe_page(client, img)
                except Exception as e:
                    transcribe_error[0] = e
            t = threading.Thread(target=_do_transcribe, daemon=True)
            t.start()
            t.join(timeout=120)
            if t.is_alive():
                log(f"  ⚠️ Page {pnum} timed out — skipping")
                arabic = f"[Page {pnum} timed out]"
            elif transcribe_error[0]:
                log(f"  ❌ Failed to transcribe page {pnum}: {transcribe_error[0]}")
                arabic = f"[Error transcribing page {pnum}]"
            else:
                arabic = transcribe_result[0]

            translation = None
            if do_translate:
                log(f"🌍 Translating page {pnum} to {target_lang.title()}...")
                try:
                    translation = translate_text(client, arabic, translation_prompt)
                except Exception as e:
                    log(f"  ❌ Failed to translate page {pnum}: {e}")
                    translation = f"[Error translating page {pnum}]"

            pages_data.append((pnum, arabic, translation))

        log("📝 Building output PDF(s)...")
        outputs = {}

        if not do_translate:
            styles = get_styles()
            story = []
            for pnum, arabic, _ in pages_data:
                add_arabic_page(story, pnum, arabic, styles)
            outputs["transcription.pdf"] = build_pdf_bytes(story)

        elif mode == "interleaved":
            styles = get_styles()
            story = []
            for pnum, arabic, trans in pages_data:
                add_arabic_page(story, pnum, arabic, styles)
                add_translation_page(story, pnum, trans, target_lang, styles)
            outputs["transcription_with_translation.pdf"] = build_pdf_bytes(story)

        elif mode == "separate":
            styles_ar = get_styles()
            ar_story = []
            for pnum, arabic, _ in pages_data:
                add_arabic_page(ar_story, pnum, arabic, styles_ar)
            outputs["transcription_arabic.pdf"] = build_pdf_bytes(ar_story)

            styles_tr = get_styles()
            tr_story = []
            for pnum, _, trans in pages_data:
                add_translation_page(tr_story, pnum, trans, target_lang, styles_tr)
            outputs[f"transcription_{target_lang}.pdf"] = build_pdf_bytes(tr_story)

        job["outputs"] = outputs
        job["status"] = "done"
        log("✅ Done!")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        log(f"❌ Fatal error: {e}")
    finally:
        if tmp_pdf and os.path.exists(tmp_pdf):
            os.unlink(tmp_pdf)
        # Clean up old jobs — keep only the 20 most recent to avoid memory leaks
        if len(jobs) > 20:
            oldest_keys = list(jobs.keys())[:-20]
            for k in oldest_keys:
                jobs.pop(k, None)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return HTML_PAGE


@app.route("/start", methods=["POST"])
def start():
    import uuid
    api_key       = request.form.get("api_key", "").strip()
    do_translate  = request.form.get("do_translate") == "true"
    target_lang   = request.form.get("target_lang", "english").strip()
    mode          = request.form.get("mode", "interleaved")
    manual_context= request.form.get("manual_context", "").strip()
    request_dpi   = request.form.get("dpi", "200")
    use_range     = request.form.get("use_range") == "true"
    start_page    = int(request.form.get("start_page", 1)) if use_range else None
    end_page      = int(request.form.get("end_page", 1))   if use_range else None

    if not api_key:
        return jsonify({"error": "API key required"}), 400

    pdf_file = request.files.get("pdf")
    if not pdf_file:
        return jsonify({"error": "PDF required"}), 400

    pdf_bytes = pdf_file.read()
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "log": [], "outputs": {}, "error": None}

    t = threading.Thread(
        target=run_job,
        args=(job_id, api_key, pdf_bytes, start_page, end_page,
              do_translate, target_lang, mode, manual_context, request_dpi),
        daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/poll/<job_id>")
def poll(job_id):
    """Simple polling endpoint — browser calls this every 2 seconds."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "log": [], "files": []})
    return jsonify({
        "status": job["status"],
        "log": job["log"],
        "files": list(job["outputs"].keys()) if job["status"] == "done" else []
    })


@app.route("/download/<job_id>/<filename>")
def download(job_id, filename):
    job = jobs.get(job_id)
    if not job or filename not in job["outputs"]:
        return "File not found", 404
    data = job["outputs"][filename]
    return send_file(
        io.BytesIO(data),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename
    )


# ── HTML (single-file app) ────────────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Matn OCR</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:      #0e0c09;
  --surface: #161210;
  --border:  #2a2218;
  --gold:    #c9a84c;
  --gold-d:  #7a6330;
  --text:    #e8dfc8;
  --muted:   #8a7d66;
}
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'EB Garamond', Georgia, serif;
  min-height: 100vh;
  display: flex;
  justify-content: center;
  padding: 0 1.5rem 4rem;
}
.wrap { width: 100%; max-width: 640px; }

/* Hero */
.hero {
  text-align: center;
  padding: 3.5rem 0 2.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2.5rem;
}
.hero-ar  { font-size: 2rem; color: var(--gold); letter-spacing: .2em; }
.hero-h1  { font-family: 'Playfair Display', serif; font-size: 3rem;
             font-weight: 700; letter-spacing: .04em; margin: .2rem 0 .5rem; }
.hero-sub { color: var(--muted); font-style: italic; font-size: 1.1rem; }

/* Section label */
.label {
  font-family: 'Playfair Display', serif;
  font-size: .7rem; letter-spacing: .22em; text-transform: uppercase;
  color: var(--gold-d); margin-bottom: .5rem; margin-top: 1.8rem;
}

/* Ornament */
.orn { text-align: center; color: var(--gold-d); margin: 1.6rem 0; letter-spacing: .6em; }

/* Inputs */
input[type=text], input[type=password], select {
  width: 100%; background: var(--surface); border: 1px solid var(--border);
  color: var(--text); border-radius: 3px; padding: .6rem .8rem;
  font-family: 'EB Garamond', serif; font-size: 1rem; outline: none;
}
input:focus, select:focus { border-color: var(--gold-d); }
.caption { color: var(--muted); font-size: .85rem; margin-top: .35rem; font-style: italic; }

/* File upload */
.file-drop {
  display: block;
  border: 1px dashed var(--border); border-radius: 3px;
  background: var(--surface); padding: 1.4rem;
  text-align: center; cursor: pointer; transition: border-color .2s;
  width: 100%; box-sizing: border-box;
}
.file-drop:hover { border-color: var(--gold-d); }
.file-drop input { display: none; }
.file-drop .icon { font-size: 1.8rem; margin-bottom: .4rem; }
.file-drop .hint { color: var(--muted); font-size: .9rem; }
.file-name { color: var(--gold); font-size: .9rem; margin-top: .4rem; }

/* Checkboxes */
.checks { display: flex; gap: 2rem; margin-top: .3rem; }
.checks label { display: flex; align-items: center; gap: .5rem;
                cursor: pointer; font-size: 1rem; }
.checks input[type=checkbox] { accent-color: var(--gold); width: 1rem; height: 1rem; }

/* Page range row */
.row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: .8rem; }
input[type=number] {
  width: 100%; background: var(--surface); border: 1px solid var(--border);
  color: var(--text); border-radius: 3px; padding: .6rem .8rem;
  font-family: 'EB Garamond', serif; font-size: 1rem; outline: none;
}

/* Button */
.btn {
  width: 100%; margin-top: 1.8rem;
  background: var(--gold); color: #0e0c09; border: none; border-radius: 3px;
  font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 700;
  letter-spacing: .08em; padding: .75rem 2rem; cursor: pointer; transition: opacity .2s;
}
.btn:hover { opacity: .85; }
.btn:disabled { opacity: .4; cursor: not-allowed; }

/* Log */
.log-wrap { margin-top: 1.5rem; display: none; }
.log-box {
  background: #070604; border: 1px solid var(--border); border-radius: 3px;
  padding: 1rem 1.1rem; font-family: 'Courier New', monospace; font-size: .82rem;
  color: #a89a7a; max-height: 300px; overflow-y: auto; line-height: 1.75;
  white-space: pre-wrap;
}

/* Downloads */
.dl-wrap { margin-top: 1.5rem; display: none; }
.dl-btn {
  display: block; width: 100%; margin-top: .6rem; padding: .65rem 1rem;
  background: transparent; border: 1px solid var(--gold-d); border-radius: 3px;
  color: var(--gold); font-family: 'EB Garamond', serif; font-size: 1rem;
  text-align: center; text-decoration: none; transition: background .2s;
}
.dl-btn:hover { background: rgba(201,168,76,.07); }

/* Error */
.err { color: #c47a7a; background: #1a0e0e; border: 1px solid #4a2020;
       border-radius: 3px; padding: .8rem 1rem; margin-top: 1rem; font-size: .95rem; }
</style>
</head>
<body>
<div class="wrap">

  <div class="hero">
    <div class="hero-ar">متن</div>
    <h1 class="hero-h1">Matn OCR</h1>
    <p class="hero-sub">Arabic manuscript transcription &amp; translation</p>
  </div>

  <form id="form" enctype="multipart/form-data">

    <div class="label">Anthropic API Key</div>
    <input type="password" name="api_key" id="api_key" placeholder="sk-ant-...">
    <p class="caption">Your key is never stored. Get one at <a href="https://console.anthropic.com" target="_blank" style="color:var(--gold-d);">console.anthropic.com</a></p>

    <div class="orn">· · ·</div>

    <div class="label">Upload PDF</div>
    <label class="file-drop" id="drop">
      <input type="file" name="pdf" id="pdf" accept=".pdf">
      <div class="icon">📜</div>
      <div class="hint">Click to browse or drag &amp; drop</div>
      <div class="file-name" id="fname"></div>
    </label>

    <div class="orn">· · ·</div>

    <div class="label">Options</div>
    <div class="checks">
      <label><input type="checkbox" id="use_range"> Specific page range</label>
      <label><input type="checkbox" id="use_trans"> Translate</label>
    </div>

    <div id="range_fields" style="display:none">
      <div class="row2">
        <div>
          <div class="label">From page</div>
          <input type="number" name="start_page" value="1" min="1">
        </div>
        <div>
          <div class="label">To page</div>
          <input type="number" name="end_page" value="10" min="1">
        </div>
      </div>
    </div>

    <div id="trans_fields" style="display:none">
      <div class="label" style="margin-top:1rem">Target language</div>
      <input type="text" name="target_lang" value="english" placeholder="english, french, urdu...">

      <div class="label">Output format</div>
      <select name="mode">
        <option value="interleaved">Arabic + Translation in one PDF</option>
        <option value="separate">Two separate PDFs</option>
      </select>

      <div class="label">Context override (optional)</div>
      <input type="text" name="manual_context" placeholder='e.g. "classical Arabic poetry" — leave blank to auto-detect'>
    </div>

    <div class="label" style="margin-top:1.2rem">Processing quality</div>
    <select name="dpi">
      <option value="200">Fast — good quality (recommended)</option>
      <option value="300">High quality — slower conversion</option>
    </select>

    <input type="hidden" name="use_range" id="h_use_range" value="false">
    <input type="hidden" name="do_translate" id="h_do_translate" value="false">

    <button type="submit" class="btn" id="submit_btn">Begin Transcription</button>
    <p class="caption" style="text-align:center; margin-top:.8rem;">
      Processing takes several minutes depending on page count — please keep this tab open.
    </p>
  </form>

  <div id="err_box" class="err" style="display:none"></div>

  <div class="log-wrap" id="log_wrap">
    <div class="label">Progress</div>
    <p class="caption" id="working_hint" style="margin-bottom:.5rem; display:none;">
      ⏳ Working — each page takes 15–20 seconds. The log will update as pages complete.
    </p>
    <div class="log-box" id="log_box"></div>
  </div>

  <div class="dl-wrap" id="dl_wrap">
    <div class="orn">· · ·</div>
    <div class="label">Download</div>
    <div id="dl_links"></div>
  </div>

  <p style="text-align:center; color:var(--muted); font-size:.85rem; font-style:italic; margin-top:3rem;">
    Questions? Contact <strong style="color:var(--gold-d);">@lumenx.sys</strong> on Discord.
  </p>

</div>

<script>
// Show/hide optional fields
document.getElementById('use_range').addEventListener('change', function() {
  document.getElementById('range_fields').style.display = this.checked ? 'block' : 'none';
  document.getElementById('h_use_range').value = this.checked ? 'true' : 'false';
});
document.getElementById('use_trans').addEventListener('change', function() {
  document.getElementById('trans_fields').style.display = this.checked ? 'block' : 'none';
  document.getElementById('h_do_translate').value = this.checked ? 'true' : 'false';
});

// Show filename when selected
document.getElementById('pdf').addEventListener('change', function() {
  document.getElementById('fname').textContent = this.files[0] ? this.files[0].name : '';
});

// Form submit
document.getElementById('form').addEventListener('submit', async function(e) {
  e.preventDefault();

  const errBox = document.getElementById('err_box');
  errBox.style.display = 'none';

  const apiKey = document.getElementById('api_key').value.trim();
  const pdfFile = document.getElementById('pdf').files[0];
  if (!apiKey) { errBox.textContent = 'Please enter your Anthropic API key.'; errBox.style.display = 'block'; return; }
  if (!pdfFile) { errBox.textContent = 'Please upload a PDF file.'; errBox.style.display = 'block'; return; }

  const btn = document.getElementById('submit_btn');
  btn.disabled = true;
  btn.textContent = 'Processing...';

  // Reset UI
  document.getElementById('log_wrap').style.display = 'block';
  document.getElementById('log_box').textContent = '';
  document.getElementById('dl_wrap').style.display = 'none';
  document.getElementById('dl_links').innerHTML = '';
  document.getElementById('working_hint').style.display = 'block';

  // Submit form
  const fd = new FormData(this);
  let jobId;
  try {
    const res = await fetch('/start', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { throw new Error(data.error); }
    jobId = data.job_id;
  } catch(err) {
    errBox.textContent = 'Failed to start: ' + err.message;
    errBox.style.display = 'block';
    btn.disabled = false;
    btn.textContent = 'Begin Transcription';
    return;
  }

  // Poll for progress every 2 seconds
  const logBox = document.getElementById('log_box');
  let lastLogCount = 0;

  async function poll() {
    try {
      const res = await fetch('/poll/' + jobId);
      const data = await res.json();

      // Append any new log lines
      for (let i = lastLogCount; i < data.log.length; i++) {
        logBox.textContent += data.log[i] + '\\n';
        logBox.scrollTop = logBox.scrollHeight;
      }
      lastLogCount = data.log.length;

      if (data.status === 'done') {
        document.getElementById('working_hint').style.display = 'none';
        btn.disabled = false;
        btn.textContent = 'Begin Transcription';
        const dlWrap = document.getElementById('dl_wrap');
        const dlLinks = document.getElementById('dl_links');
        dlWrap.style.display = 'block';
        data.files.forEach(function(fname) {
          const a = document.createElement('a');
          a.href = '/download/' + jobId + '/' + encodeURIComponent(fname);
          a.className = 'dl-btn';
          a.textContent = '⬇ Download ' + fname;
          a.download = fname;
          dlLinks.appendChild(a);
        });
        return; // stop polling
      }

      if (data.status === 'error') {
        document.getElementById('working_hint').style.display = 'none';
        btn.disabled = false;
        btn.textContent = 'Begin Transcription';
        return; // stop polling
      }

      // Still running — poll again in 2 seconds
      setTimeout(poll, 2000);
    } catch(err) {
      // Network hiccup — retry in 3 seconds
      setTimeout(poll, 3000);
    }
  }

  poll();
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
