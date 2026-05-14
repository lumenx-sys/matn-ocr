import streamlit as st
import os
import base64
import time
import tempfile
import threading
import queue
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
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Matn OCR",
    page_icon="📜",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap');

:root {
    --bg:       #0e0c09;
    --surface:  #181410;
    --border:   #2e2820;
    --gold:     #c9a84c;
    --gold-dim: #7a6330;
    --text:     #e8dfc8;
    --muted:    #8a7d66;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'EB Garamond', Georgia, serif;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }

.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.hero-arabic {
    font-size: 2.2rem;
    color: var(--gold);
    letter-spacing: 0.15em;
    margin-bottom: 0.3rem;
    font-family: 'Playfair Display', serif;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.05em;
    margin: 0;
}
.hero-sub {
    color: var(--muted);
    font-style: italic;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}
.section-label {
    font-family: 'Playfair Display', serif;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold-dim);
    margin-bottom: 0.5rem;
}
.ornament {
    text-align: center;
    color: var(--gold-dim);
    font-size: 1.2rem;
    margin: 1.5rem 0;
    letter-spacing: 0.5em;
}
.log-box {
    background: #080705;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #a89a7a;
    max-height: 280px;
    overflow-y: auto;
    line-height: 1.7;
    white-space: pre-wrap;
}
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select,
[data-testid="stNumberInput"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
}
.stButton > button {
    background: var(--gold) !important;
    color: #0e0c09 !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: var(--gold) !important;
    border: 1px solid var(--gold-dim) !important;
    border-radius: 3px !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 1rem !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Font registration (cached, runs once per server session) ──────────────────
@st.cache_resource
def register_arabic_font():
    """Register a Unicode-capable font for Arabic PDF output."""
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

    # Fallback: download Amiri (a proper Arabic font) from a reliable CDN
    import urllib.request
    font_path = "/tmp/arabic_font.ttf"
    if not os.path.exists(font_path):
        url = "https://fonts.gstatic.com/s/amiri/v27/J7aRnpd8CGxBHqUpvrIw74NL.woff2"
        # woff2 won't work for reportlab — use a TTF CDN instead
        url = "https://github.com/alif-type/amiri/releases/download/1.000/Amiri-1.000.zip"
        # Simpler: just grab the TTF directly from jsDelivr
        url = "https://cdn.jsdelivr.net/npm/@fontsource/amiri@5.0.8/files/amiri-arabic-400-normal.ttf"
        try:
            urllib.request.urlretrieve(url, font_path)
            pdfmetrics.registerFont(TTFont("Arabic", font_path))
            return
        except Exception:
            pass

    # Last resort: use Helvetica (limited Arabic support but won't crash)
    pdfmetrics.registerFont(TTFont("Arabic", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))


register_arabic_font()


# ── PDF styles (defined after font registration) ──────────────────────────────
def get_styles():
    base = getSampleStyleSheet()
    heading = ParagraphStyle("MatnHeading", parent=base["Heading2"], spaceAfter=6)
    arabic  = ParagraphStyle("MatnArabic",  parent=base["Normal"],
                              fontName="Arabic", fontSize=12, leading=22,
                              spaceAfter=4, alignment=2)
    trans   = ParagraphStyle("MatnTrans",   parent=base["Normal"],
                              fontSize=11, leading=18, spaceAfter=4, alignment=0)
    return heading, arabic, trans


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


# ── Core functions ────────────────────────────────────────────────────────────
def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def call_claude(client, log_q, **kwargs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                log_q.put(f"  ⚠ Error (attempt {attempt+1}/{max_retries}): {e}")
                log_q.put(f"  ⏳ Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise e


def detect_context(client, img, log_q):
    log_q.put("🔍 Auto-detecting book context from first page...")
    img_b64 = image_to_base64(img)
    msg = call_claude(client, log_q,
        model="claude-sonnet-4-6", max_tokens=300,
        system=DETECT_PROMPT,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/jpeg", "data": img_b64}},
            {"type": "text", "text": "What is the context of this text?"}
        ]}]
    )
    context = msg.content[0].text.strip()
    log_q.put(f"  📖 Detected: {context}\n")
    return context


def transcribe_page(client, img, log_q):
    img_b64 = image_to_base64(img)
    msg = call_claude(client, log_q,
        model="claude-sonnet-4-6", max_tokens=4096,
        system=TRANSCRIBE_PROMPT,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/jpeg", "data": img_b64}},
            {"type": "text", "text": "Please transcribe all Arabic text on this page exactly as it appears."}
        ]}]
    )
    return msg.content[0].text


def translate_page(client, arabic_text, translation_prompt, log_q):
    msg = call_claude(client, log_q,
        model="claude-sonnet-4-6", max_tokens=4096,
        system=translation_prompt,
        messages=[{"role": "user", "content": arabic_text}]
    )
    return msg.content[0].text


def fix_arabic(text):
    return get_display(arabic_reshaper.reshape(text))


def sanitize(text):
    """Replace special characters that basic fonts cannot render."""
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2014": "-", "\u2013": "-",
        "\u2026": "...",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("ascii", errors="ignore").decode("ascii")


def escape_html(text):
    """Escape text for safe insertion into HTML log box."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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


def run_ocr(api_key, pdf_bytes, start_page, end_page,
            do_translate, target_lang, mode, manual_context,
            log_q, result_q):
    tmp_pdf = None
    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Write PDF to temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_pdf = f.name

        # Convert to images
        log_q.put("🔄 Converting PDF pages to images...")
        conv_kwargs = dict(dpi=300)
        if start_page:
            conv_kwargs["first_page"] = start_page
        if end_page:
            conv_kwargs["last_page"] = end_page
        images = convert_from_path(tmp_pdf, **conv_kwargs)
        base_page = start_page if start_page else 1
        page_numbers = list(range(base_page, base_page + len(images)))
        log_q.put(f"✅ {len(images)} pages ready\n")

        # Detect context if translating
        translation_prompt = None
        if do_translate:
            if manual_context and manual_context.strip():
                context = manual_context.strip()
                log_q.put(f"📖 Using manual context: {context}\n")
            else:
                context = detect_context(client, images[0], log_q)
            translation_prompt = build_translation_prompt(target_lang, context)

        # Process each page
        pages_data = []
        for i, (img, pnum) in enumerate(zip(images, page_numbers)):
            log_q.put(f"🤖 Transcribing page {pnum} ({i+1}/{len(images)})...")
            try:
                arabic = transcribe_page(client, img, log_q)
            except Exception as e:
                log_q.put(f"  ❌ Failed to transcribe: {e}")
                arabic = f"[Error transcribing page {pnum}: {e}]"

            translation = None
            if do_translate:
                log_q.put(f"🌍 Translating page {pnum} to {target_lang.title()}...")
                try:
                    translation = translate_page(client, arabic, translation_prompt, log_q)
                except Exception as e:
                    log_q.put(f"  ❌ Failed to translate: {e}")
                    translation = f"[Error translating page {pnum}: {e}]"

            pages_data.append((pnum, arabic, translation))

        # Build output PDFs — get fresh styles for each PDF to avoid name conflicts
        log_q.put("\n📝 Building output PDF(s)...")
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

        log_q.put("✅ Done!")
        result_q.put(("ok", outputs))

    except Exception as e:
        log_q.put(f"\n❌ Fatal error: {e}")
        result_q.put(("error", str(e)))
    finally:
        # Always clean up temp file
        if tmp_pdf and os.path.exists(tmp_pdf):
            os.unlink(tmp_pdf)


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-arabic">متن</div>
    <div class="hero-title">Matn OCR</div>
    <div class="hero-sub">Arabic manuscript transcription &amp; translation</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Anthropic API Key</div>', unsafe_allow_html=True)
api_key = st.text_input("", type="password", placeholder="sk-ant-...",
                         label_visibility="collapsed")
st.caption("Your key is never stored. Get one at console.anthropic.com")

st.markdown('<div class="ornament">❧ ❧ ❧</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Upload PDF</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

st.markdown('<div class="ornament">❧ ❧ ❧</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Options</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    use_page_range = st.checkbox("Specific page range")
with col2:
    do_translate = st.checkbox("Translate")

start_page = end_page = None
if use_page_range:
    c1, c2 = st.columns(2)
    with c1:
        start_page = st.number_input("From page", min_value=1, value=1, step=1)
    with c2:
        end_page = st.number_input("To page", min_value=1, value=10, step=1)

target_lang = mode = manual_context = None
if do_translate:
    target_lang = st.text_input("Target language", value="english",
                                 placeholder="english, french, urdu...")
    mode = st.selectbox(
        "Output format", ["interleaved", "separate"],
        format_func=lambda x: "Arabic + Translation in one PDF"
                               if x == "interleaved" else "Two separate PDFs"
    )
    manual_context = st.text_input(
        "Context override (optional)",
        placeholder='e.g. "classical Arabic poetry" — leave blank to auto-detect'
    )

st.markdown('<div class="ornament">❧ ❧ ❧</div>', unsafe_allow_html=True)

run = st.button("Begin Transcription")

if run:
    if not api_key:
        st.error("Please enter your Anthropic API key.")
    elif not uploaded_file:
        st.error("Please upload a PDF file.")
    elif do_translate and not target_lang:
        st.error("Please enter a target language.")
    else:
        pdf_bytes = uploaded_file.read()
        log_q = queue.Queue()
        result_q = queue.Queue()

        t = threading.Thread(
            target=run_ocr,
            args=(api_key, pdf_bytes,
                  int(start_page) if use_page_range and start_page else None,
                  int(end_page) if use_page_range and end_page else None,
                  do_translate, target_lang, mode, manual_context,
                  log_q, result_q),
            daemon=True
        )
        t.start()

        st.markdown('<div class="section-label">Progress</div>', unsafe_allow_html=True)
        log_placeholder = st.empty()
        log_lines = []

        while t.is_alive() or not log_q.empty():
            changed = False
            while not log_q.empty():
                log_lines.append(escape_html(log_q.get()))
                changed = True
            if changed:
                log_placeholder.markdown(
                    '<div class="log-box">' + "\n".join(log_lines) + '</div>',
                    unsafe_allow_html=True
                )
            time.sleep(0.3)

        # Final flush
        while not log_q.empty():
            log_lines.append(escape_html(log_q.get()))
        log_placeholder.markdown(
            '<div class="log-box">' + "\n".join(log_lines) + '</div>',
            unsafe_allow_html=True
        )

        if not result_q.empty():
            status, payload = result_q.get()
            if status == "ok":
                st.markdown('<div class="ornament">❧ ❧ ❧</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Download</div>',
                            unsafe_allow_html=True)
                for filename, data in payload.items():
                    st.download_button(
                        label=f"⬇ Download {filename}",
                        data=data,
                        file_name=filename,
                        mime="application/pdf"
                    )
            else:
                st.error(f"Something went wrong: {payload}")
