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
from pdf2image.pdf2image import pdfinfo_from_path
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
Your only job is to transcribe the Arabic text faithfully, preserving spelling and orthography as printed.
Rules:
- Transcribe the Arabic text EXACTLY as it appears, preserving spelling and orthography
- Preserve diacritics (harakat) only when clearly visible in the image — do not guess or add them
- You may silently normalize obvious broken letter shapes into correct Arabic words when unambiguous
- Preserve line breaks and punctuation as they appear
- If a word is genuinely unclear or damaged beyond recognition, write [?] in its place
- Do not add commentary, explanations, or notes
- Do not translate anything
- Output only the transcribed text, nothing else"""

DETECT_PROMPT = """You are an expert in Arabic literature and Islamic sciences.
Look at this Arabic text and identify the book's context in 1-2 sentences.
Describe: the genre, subject matter, scholarly tradition, time period, and any relevant terminology conventions a translator should know.
Be specific and concise. Output only the description, nothing else."""


def build_translation_prompt(target_language, context):
    return f"""You are translating classical Arabic Islamic legal texts into clear, scholarly {target_language}.
The text you are translating is: {context}

TRANSLATION PHILOSOPHY — Your goals in order:
1. Preserve the legal meaning accurately
2. Produce natural, readable {target_language} that an educated reader studying Islamic law can follow
3. Translate technical terms whenever established {target_language} equivalents exist
4. Use transliteration only for genuinely untranslatable technical vocabulary
5. Resolve obvious OCR corruption from context and standard fiqh phrasing
6. Preserve the concise style of the text without sounding robotic

TRANSLATION STYLE:
- Write like a modern academic translator of classical Islamic texts
- Prioritize clarity and legal intelligibility over word-for-word literalism
- Avoid archaic pseudo-biblical English ("he doth", "thereof", etc.)
- Avoid preserving Arabic sentence order when it sounds unnatural in {target_language}
- The result should read like professionally translated Islamic legal literature, not a raw word-for-word gloss
- Translate for an educated {target_language} reader studying Islamic law, not for someone reading a word-for-word interlinear gloss

WHAT TO TRANSLATE (do NOT leave these in Arabic):
- wudu / وضوء → ablution
- ghusl / غسل → ritual bath
- salah / صلاة → prayer
- sawm / صيام → fasting
- hajj → pilgrimage (but keep "Hajj" as a proper noun for the rite)
- niyyah / نية → intention
- najasah / نجاسة → ritual impurity
- taharah / طهارة → purification / ritual purity
- fard / فرض → obligatory
- sunnah → recommended (when used as a legal category, not the Prophet's practice)
- mustahabb → recommended / desirable
- makruh / مكروه → disliked
- mubah / مباح → permissible
- yubtal / يبطل → invalidates
- yujzi / يجزئ → suffices
- farj / فرج → private parts (or "sexual organ" depending on context)
- jawf → body cavity
- التقاء الختانين → sexual intercourse
- ما لا نفس له سائلة → creatures without flowing blood
- ماء السماء → rainwater
- ماء البحر → seawater
- ماء النهر → river water
- muhdith → a person in a state of minor ritual impurity
- junub → a person in a state of major ritual impurity
- wali → guardian (in nikah context)
- qadi → judge
- imam → leader / prayer leader (context dependent)
- diyah → blood money
- hadd / hudud → prescribed punishment(s)
- qisas → retaliation
- nikah → marriage contract
- talaq → divorce
- iddah → waiting period
- nafaqah → financial support / maintenance

WHAT TO KEEP IN TRANSLITERATION (no standard {target_language} equivalent):
- ihram, talbiyah, tawaf, sa'y, wuquf, miqat (Hajj rites)
- qiblah, adhan, iqamah (prayer direction/call)
- zakat, nisab, hawl (zakat terms)
- ijtihad, qiyas, ijma (legal methodology)
- matn, fiqh, madhhab, fatwa (scholarly terms)
- Shafi'i, Hanbali, Maliki, Hanafi (school names)
- proper names: Ibn, Abu, Abd, Sheikh, al- prefixes
- specific technical terms with no clean equivalent: tayammum, siwak, khuff

TRANSLITERATION FORMAT:
- When keeping a term in transliteration, you may add its {target_language} meaning in parentheses on first use only
- Example: "tayammum (dry ablution)" on first use, then just "tayammum" thereafter
- Use ONLY plain ASCII characters — no diacritical marks, no curly quotes
- Standardize spelling throughout: always "Shafi'i" not "Shafiqi", "al-Asfahani" not "al-Asfehani"

OCR CORRECTION:
- The Arabic source may contain OCR errors marked as [?] or corrupted text
- Silently correct obvious errors using context and your knowledge of standard fiqh texts
- Do NOT reproduce [?] markers or corrupted transliterations in the output
- Prefer canonical readings from well-known fiqh texts

EXAMPLES OF DESIRED OUTPUT:
Bad: "The waters by which purification is permitted are seven waters"
Good: "There are seven types of water that may be used for purification"

Bad: "the meeting of the two circumcised parts"
Good: "sexual intercourse"

Bad: "that which has no flowing soul"
Good: "creatures without flowing blood"

Bad: "water of the sky"
Good: "rainwater"

Bad: "the fara'id of ghusl are three things: the niyyah..."
Good: "The obligatory acts of the ritual bath are three: the intention..."

Bad: "And the hay'ah — he does not return to it..."
Good: "As for the recommended postures — he does not return to them..."

STYLE TARGET:
Translate into fluent, modern academic English in the style of scholarly works on Islamic law (e.g. Brill Islamic law series, Oxford Islamic legal studies). Use clear, precise legal phrasing — not literal gloss translation.

ANTI-LITERALISM RULE:
Do not translate Arabic expressions word-for-word when a natural English legal equivalent exists. Rephrase idioms and formulaic legal expressions naturally.

CONSISTENCY RULE:
Maintain consistent terminology throughout the entire text. Do not alternate between different renderings of the same term.

NO ARABIC SCRIPT:
Do not include Arabic script in the output.

OUTPUT:
- Output only the translation, nothing else
- No commentary, footnotes, or explanations
- Preserve paragraph breaks as in the original"""


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


def post_process_translation(client, pages_data, target_lang):
    """Run a final pass over all translated pages to standardize terminology."""
    full_text = ""
    for pnum, arabic, translation in pages_data:
        if translation and not translation.startswith("["):
            full_text += "=== Page " + str(pnum) + " ===\n" + translation + "\n\n"

    if not full_text.strip():
        return pages_data

    system_prompt = (
        "You are a professional editor reviewing a translated Islamic legal text in "
        + target_lang + ".\n"
        "Your only job is to standardize inconsistent terminology throughout the document.\n\n"
        "Rules:\n"
        "- Identify any terms translated inconsistently and standardize them to the most accurate rendering\n"
        "- Do NOT change the meaning, structure, or content of any sentence\n"
        "- Do NOT add or remove any text\n"
        "- Keep all page markers (=== Page N ===) exactly as they are\n"
        "- Output the full corrected text with the same page markers, nothing else"
    )

    msg = call_claude(client,
        model="claude-sonnet-4-6",
        max_tokens=16000,
        system=system_prompt,
        messages=[{"role": "user", "content": "Please standardize the terminology in this translated text:\n\n" + full_text}]
    )

    corrected = msg.content[0].text.strip()

    # Parse corrected text back into per-page translations
    corrected_map = {}
    current_page = None
    current_lines = []
    for line in corrected.split("\n"):
        if line.startswith("=== Page ") and line.endswith(" ==="):
            if current_page is not None:
                corrected_map[current_page] = "\n".join(current_lines).strip()
            try:
                current_page = int(line.replace("=== Page ", "").replace(" ===", ""))
            except ValueError:
                current_page = None
            current_lines = []
        else:
            current_lines.append(line)
    if current_page is not None:
        corrected_map[current_page] = "\n".join(current_lines).strip()

    # Update pages_data with corrected translations
    updated = []
    for pnum, arabic, translation in pages_data:
        if pnum in corrected_map and not (translation or "").startswith("["):
            updated.append((pnum, arabic, corrected_map[pnum]))
        else:
            updated.append((pnum, arabic, translation))
    return updated



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
        BATCH_SIZE = 5  # convert and process this many pages at a time

        # Get total page count first using pdfinfo (very fast, no image conversion)
        info = pdfinfo_from_path(tmp_pdf)
        total_pages = info["Pages"]
        first = start_page if start_page else 1
        last  = end_page   if end_page   else total_pages
        all_page_numbers = list(range(first, last + 1))
        total = len(all_page_numbers)
        log(f"📥 PDF received — {total} pages to process")
        log(f"🔄 Processing in batches of {BATCH_SIZE} pages (quality: {'high' if dpi == 300 else 'fast'})...")

        # Auto-detect context from first page before batching
        translation_prompt = None
        if do_translate:
            if manual_context and manual_context.strip():
                context = manual_context.strip()
                log(f"📖 Using manual context: {context}")
            else:
                log("🔍 Auto-detecting book context from first page...")
                first_images = convert_from_path(tmp_pdf, dpi=dpi, first_page=first, last_page=first)
                img_b64 = image_to_base64(first_images[0])
                del first_images  # free memory immediately
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

        # Process in batches
        pages_data = []
        processed = 0
        for batch_start in range(0, total, BATCH_SIZE):
            batch_pages = all_page_numbers[batch_start:batch_start + BATCH_SIZE]
            batch_first = batch_pages[0]
            batch_last  = batch_pages[-1]

            log(f"🔄 Converting pages {batch_first}–{batch_last} to images...")
            batch_images = convert_from_path(tmp_pdf, dpi=dpi,
                                             first_page=batch_first, last_page=batch_last)

            for img, pnum in zip(batch_images, batch_pages):
                processed += 1
                log(f"🤖 Transcribing page {pnum} ({processed}/{total})...")

                arabic = None
                for attempt in range(5):
                    transcribe_result = [None]
                    transcribe_error  = [None]
                    def _do_transcribe(img=img):
                        try:
                            transcribe_result[0] = transcribe_page(client, img)
                        except Exception as e:
                            transcribe_error[0] = e
                    t = threading.Thread(target=_do_transcribe, daemon=True)
                    t.start()
                    t.join(timeout=120)
                    if t.is_alive():
                        wait = 30 * (attempt + 1)
                        log(f"  ⚠️ Transcription attempt {attempt+1}/5 timed out")
                        log(f"  ⏳ Retrying in {wait}s...")
                        time.sleep(wait)
                    elif transcribe_error[0]:
                        wait = 30 * (attempt + 1)
                        log(f"  ⚠️ Transcription attempt {attempt+1}/5 failed: {transcribe_error[0]}")
                        log(f"  ⏳ Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        arabic = transcribe_result[0]
                        break
                if arabic is None:
                    arabic = f"[Failed to transcribe page {pnum} after 5 attempts]"
                    log(f"  ❌ Giving up on page {pnum} transcription after 5 attempts")

                translation = None
                if do_translate and not arabic.startswith("[Failed"):
                    log(f"🌍 Translating page {pnum} to {target_lang.title()}...")
                    for attempt in range(5):
                        translate_result = [None]
                        translate_error  = [None]
                        def _do_translate(arabic=arabic):
                            try:
                                translate_result[0] = translate_text(client, arabic, translation_prompt)
                            except Exception as e:
                                translate_error[0] = e
                        tt = threading.Thread(target=_do_translate, daemon=True)
                        tt.start()
                        tt.join(timeout=120)
                        if tt.is_alive():
                            wait = 30 * (attempt + 1)
                            log(f"  ⚠️ Translation attempt {attempt+1}/5 timed out")
                            log(f"  ⏳ Retrying in {wait}s...")
                            time.sleep(wait)
                        elif translate_error[0]:
                            wait = 30 * (attempt + 1)
                            log(f"  ⚠️ Translation attempt {attempt+1}/5 failed: {translate_error[0]}")
                            log(f"  ⏳ Retrying in {wait}s...")
                            time.sleep(wait)
                        else:
                            translation = translate_result[0]
                            break
                    if translation is None:
                        translation = f"[Failed to translate page {pnum} after 5 attempts]"
                        log(f"  ❌ Giving up on page {pnum} translation after 5 attempts")

                pages_data.append((pnum, arabic, translation))

        # Post-processing pass to standardize terminology (only when translating)
        if do_translate and len(pages_data) > 1:
            log("✨ Running post-processing pass to standardize terminology...")
            try:
                pages_data = post_process_translation(client, pages_data, target_lang)
                log("✅ Terminology standardized")
            except Exception as e:
                log(f"  ⚠️ Post-processing failed (continuing without it): {e}")

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

    <div style="margin-top:1.2rem; background:#111009; border:1px solid var(--border); border-radius:3px; padding:1rem 1.2rem; font-size:.88rem; color:var(--muted); line-height:1.8;">
      <strong style="color:var(--gold-d); letter-spacing:.1em; font-size:.75rem; text-transform:uppercase;">Usage Guide</strong><br><br>
      <strong style="color:var(--text);">Page limit:</strong> For best results, process a maximum of <strong style="color:var(--text);">50 pages at a time</strong> using the page range option. Larger books should be done in batches (e.g. pages 1–50, then 51–100, etc.).<br><br>
      <strong style="color:var(--text);">Recommended:</strong> Translate <strong style="color:var(--text);">one chapter at a time</strong> whenever possible. Each chapter covers a single topic (e.g. purification, prayer, zakat), which helps the translation stay consistent and accurate — Claude maintains better terminology choices within a focused subject area than across many unrelated chapters.<br><br>
      <strong style="color:var(--text);">Cost estimate:</strong> Transcription only ~$0.024/page · With translation ~$0.05/page<br><br>
      <strong style="color:var(--text);">API key:</strong> Get one at <a href="https://console.anthropic.com" target="_blank" style="color:var(--gold-d);">console.anthropic.com</a> — you only pay for what you use, no subscription needed. A $5 top-up covers hundreds of pages.
    </div>
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
