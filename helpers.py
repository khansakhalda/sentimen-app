import os, re, glob, inspect, sys, subprocess
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
# from wordcloud import WordCloud
import importlib.util

# KONSTANTA
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

BASE_SIMPLE = "app_source_getcontact"
DEFAULT_PAGE_SIZE = 200
DEFAULT_TABLE_PAGE_SIZE = 20

CSS = """
<style>
[data-testid="stSidebar"] {background:#e5e7eb;}
h1,h2,h3 {font-weight:700;}
.label-pos{color:#16a34a;font-weight:700;}
.label-neg{color:#dc2626;font-weight:700;}
.kpill{display:inline-block;padding:.25rem .6rem;border-radius:999px;background:#eef2ff;border:1px solid #c7d2fe}
div[data-testid="stMetric"] > label p { font-size: 0.9rem !important; line-height: 1.1 !important; }
div[data-testid="stMetricValue"] { font-size: 1.9rem !important; line-height: 1.1 !important; white-space: nowrap !important; }
</style>
"""

def setup_page(title: str):
    st.set_page_config(page_title=title, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

# STATE
def init_state():
    ss = st.session_state
    ss.setdefault("page", "Deskripsi Analisis")
    ss.setdefault("pkg", "app.source.getcontact")
    ss.setdefault("res_collect", None)
    ss.setdefault("res_clean", None)
    ss.setdefault("res_casefold", None)
    ss.setdefault("res_normalize", None)
    ss.setdefault("res_tokenize", None)
    ss.setdefault("res_label", None)
    ss.setdefault("res_tfidf_feat", None)
    ss.setdefault("res_tfidf_top5", None)
    ss.setdefault("res_split", None)
    ss.setdefault("res_svm_linear", None)
    ss.setdefault("res_svm_rbf", None)
    ss.setdefault("rbf_subsample_frac", 0.15)

def restart():
    keys = list(st.session_state.keys())
    for k in keys: del st.session_state[k]
    st.rerun()

# IMPORT PIPELINE
def import_fix_module() -> Tuple[object, Optional[str]]:
    """Muat modul pipeline (pipeline.py)."""
    for path in ["pipeline.py", os.path.join("data", "pipeline.py")]:
        if os.path.exists(path):
            try:
                spec = importlib.util.spec_from_file_location("fixmod", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                required = ["stage1_collect","stage6_label_inset","stage7_tfidf",
                            "stage8_split","stage9_svm_linear","stage10_svm_rbf"]
                for fn in required:
                    if not hasattr(mod, fn):
                        st.warning(f"Modul {os.path.basename(path)} tidak punya fungsi wajib: {fn}")
                return mod, path
            except Exception as e:
                st.exception(e)
                st.error(f"Gagal import: {path}")
                return None, path  # type: ignore
    return None, None  # type: ignore

# STOPWORDS
@st.cache_resource
def get_stopwords_id() -> set:
    """Ambil stopwords Indonesia otomatis. Urutan: NLTK → Sastrawi → set()."""
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            return set(stopwords.words("indonesian"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("indonesian"))
    except Exception:
        pass
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "Sastrawi"])
            from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        except Exception:
            return set()
    factory = StopWordRemoverFactory()
    return set(factory.get_stop_words())

_warned_no_stop = False

# UTIL DATA/UI
def tok_simple(s: str):
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", str(s).lower())

def top_words(series: pd.Series, n: int = 20):
    global _warned_no_stop
    stop_id = get_stopwords_id()
    if not stop_id and not _warned_no_stop:
        st.warning("Stopwords Bahasa Indonesia tidak tersedia (NLTK/Sastrawi). Menampilkan tanpa filter stopwords.")
        _warned_no_stop = True
    cnt = Counter()
    for s in series.fillna("").astype(str):
        for w in tok_simple(s):
            if (w not in stop_id) and len(w) >= 3:
                cnt[w] += 1
    return cnt.most_common(n)

def auto_insight_from_raw(df_raw: pd.DataFrame):
    tips = []
    at_min = at_max = None
    if "at" in df_raw.columns:
        at_ser = pd.to_datetime(df_raw["at"], errors="coerce")
        at_min = at_ser.min(); at_max = at_ser.max()
    n_total = int(len(df_raw))
    if pd.notna(at_min) and pd.notna(at_max):
        days = max(1, (at_max - at_min).days + 1)
        dens = n_total / days
        tips.append(f"Rentang data {days} hari ({at_min:%d %b %y} – {at_max:%d %b %y}); ±{dens:,.0f} ulasan/hari.")
        if days < 14:
            tips.append("Rentang < 2 minggu; hasil bisa bias periode pendek. Pertimbangkan perluas periode.")
    else:
        tips.append("Kolom waktu (‘at’) sebagian tidak valid; analisis time-series terbatas.")
    if "content" in df_raw.columns:
        try:
            avg_len = float(df_raw["content"].astype(str).map(lambda s: len(tok_simple(s))).mean())
            tips.append(f"Panjang ulasan rata-rata ≈ {avg_len:.0f} kata.")
            if avg_len < 3:
                tips.append("Rata-rata sangat pendek (<3 kata). Pertimbangkan filter minimal panjang/anti-spam.")
        except Exception:
            pass
    if "score" in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw["score"]):
        _ = float(df_raw["score"].mean())
        _ = df_raw["score"].value_counts().reindex([1,2,3,4,5], fill_value=0)
    return tips[:8] if tips else ["Tidak ada temuan pada tahap pengumpulan."]

def latest(pattern: str) -> Optional[str]:
    f = sorted(glob.glob(pattern))
    return f[-1] if f else None

def load_csv(p: str) -> pd.DataFrame:
    return pd.read_csv(p, encoding="utf-8-sig")

def multi_match_mask(series: pd.Series, terms, whole_word: bool, mode_all: bool) -> pd.Series:
    def _word_regex(t):
        t = re.escape(t.strip())
        return rf"(?i)(?:^|\b){t}(?:\b|$)" if whole_word else rf"(?i){t}"
    masks = [series.astype(str).str.contains(_word_regex(t), na=False) for t in terms]
    if not masks: return pd.Series([True]*len(series), index=series.index)
    return np.logical_and.reduce(masks) if mode_all else np.logical_or.reduce(masks)

def btn(label: str, key: Optional[str] = None) -> bool:
    return st.button(label, key=key)

def offer_download(path: str, label: str):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            st.download_button(label, f.read(), file_name=os.path.basename(path))

def show_images_side_by_side(paths: List[str], captions: List[str], width: Optional[int]=None):
    cols = st.columns(len(paths))
    for i, p in enumerate(paths):
        if not p:
            continue

        safe_path = os.path.abspath(p).replace("\\", "/")

        if os.path.exists(safe_path):
            cols[i].image(safe_path, caption=captions[i], width=width)
        else:
            cols[i].warning(f"Gambar tidak ditemukan: {safe_path}")

def build_wc_and_freq_before(csv_path: str, wc_path: str, freq_path: str):
    df_raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "content" not in df_raw.columns:
        return None, None

    texts = df_raw["content"].fillna("").astype(str).tolist()
    joined = "\n".join(texts)

    # # WordCloud
    # wc = WordCloud(width=1000, height=600, background_color="white").generate(joined)
    # fig1, ax1 = plt.subplots(figsize=(9, 5))
    # ax1.imshow(wc)
    # ax1.axis("off")
    # fig1.tight_layout()
    # fig1.savefig(wc_path, dpi=120, bbox_inches="tight")
    # plt.close(fig1)

    # Frekuensi
    from collections import Counter
    import re
    tokens = []
    for t in texts:
        tokens += re.findall(r"\b\w+\b", str(t).lower())
    freqs = Counter(tokens).most_common(15)

    if freqs:
        words, counts = zip(*freqs)
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        y_pos = list(range(len(words)))[::-1]
        vals = list(counts)[::-1]

        ax2.barh(y_pos, vals)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(words[::-1])
        ax2.set_xlabel("Frekuensi")
        ax2.set_title("Top-15 Frekuensi Kata (sebelum preprocessing)")

        def indo(n):
            return f"{n:,}".replace(",", ".")

        offset = max(vals) * 0.02
        for y, v in zip(y_pos, vals):
            ax2.text(v + offset, y, indo(v), va="center", fontsize=10)

        fig2.tight_layout()
        fig2.savefig(freq_path, dpi=120, bbox_inches="tight")
        plt.close(fig2)

    return wc_path, freq_path


def build_wc_and_freq_after(texts: List[str], wc_path: str, freq_path: str):
    joined = " ".join(texts)

    # # ===== WordCloud =====
    # wc = WordCloud(width=1000, height=500, background_color="white").generate(joined)
    # fig1, ax1 = plt.subplots(figsize=(9, 4))
    # ax1.imshow(wc)
    # ax1.axis("off")
    # fig1.tight_layout()
    # fig1.savefig(wc_path, dpi=120, bbox_inches="tight")
    # plt.close(fig1)

    # ===== Bar chart frekuensi =====
    words = re.findall(r"\b[a-z]{3,}\b", joined.lower())
    top = Counter(words).most_common(15)

    if top:
        wds, cts = zip(*top)

        fig2, ax2 = plt.subplots(figsize=(9, 5))

        y_pos = list(range(len(wds)))[::-1]
        vals  = list(cts)[::-1]

        ax2.barh(y_pos, vals)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(wds[::-1])
        ax2.set_xlabel("Frekuensi")
        ax2.set_title("Top-15 Frekuensi Kata (setelah preprocessing)")

        # ===== FORMAT ANGKA INDONESIA =====
        def format_indo(n: int) -> str:
            return f"{n:,}".replace(",", ".")

        offset = max(cts) * 0.02  
        for y, v in zip(y_pos, vals):
            ax2.text(v + offset, y, format_indo(v), va="center", fontsize=10)

        fig2.tight_layout()
        fig2.savefig(freq_path, dpi=120, bbox_inches="tight")
        plt.close(fig2)

def load_slang_map(path="slangs.txt"):
    slang = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or ":" not in line: continue
                k, v = line.split(":", 1)
                k = k.strip().lower(); v = v.strip().lower()
                if k: slang[k] = v
    return slang

def normalize_with_slang(text: str, slang_map: dict) -> str:
    """Normalisasi slang: frasa dulu, lalu per-kata."""
    if not isinstance(text, str) or not text:
        return ""
    text = re.sub(r"\s+", " ", text.lower()).strip()
    if not slang_map:
        return text
    phrase_keys = [k for k in slang_map.keys() if " " in k]
    for k in sorted(phrase_keys, key=len, reverse=True):
        v = slang_map[k]
        pat = rf"\b{re.escape(k)}\b"
        text = re.sub(pat, v, text)
    tokens = text.split()
    tokens = [slang_map.get(tok, tok) for tok in tokens]
    return " ".join(tokens).strip()

def has_param(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except Exception:
        return False

# UI Helper: Deskripsi Tiap Seksi
def section_desc(text: str, expanded: bool = True, title: str = "ℹ️ Deskripsi singkat"):
    """Kotak deskripsi ringkas yang bisa dibuka-tutup (default: terbuka)."""
    with st.expander(title, expanded=expanded):
        st.markdown(text)

# download
def make_download_button(path: str, label: str, filename: str) -> str:
    """Generate HTML tombol download custom sesuai style aplikasi."""
    import base64, os

    if not os.path.exists(path):
        return "<p style='color:red;'>File tidak ditemukan.</p>"

    with open(path, "rb") as f:
        file_data = f.read()
        b64 = base64.b64encode(file_data).decode()

    return f"""
        <style>
            .download-btn {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 8px 18px;
                background-color: #ffffff;
                border: 1px solid #d6d6d6;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 500;
                color: #000000;
                text-decoration: none !important;
                cursor: pointer;
            }}
            .download-btn:hover {{
                background-color: #f3f4f6;
                border-color: #c0c0c0;
            }}
        </style>

        <a href="data:file/csv;base64,{b64}"
           download="{filename}"
           class="download-btn">
            <span style="color:#4da3ff; font-size:18px;">⬇️</span>
            {label}
        </a>
    """

