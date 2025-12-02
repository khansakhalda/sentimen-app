import os
import io
import re
import pandas as pd
import streamlit as st
import altair as alt
from collections import Counter
from itertools import tee
from helpers import load_csv, top_words, load_slang_map, normalize_with_slang, latest, DATA_DIR

from dotenv import load_dotenv
load_dotenv()

# INIT SESSION CACHE UNTUK GEMINI SUMMARY
if "summary_cache" not in st.session_state:
    st.session_state["summary_cache"] = {}

# GOOGLE GENAI API (VERSI BARU)
from google import genai
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# KONSTANTA
TITLE = "Analisis Tren Sentimen (Triwulan)" 
PERIOD_FROM = pd.Timestamp("2024-09-01")
PERIOD_TO   = pd.Timestamp("2025-08-31 23:59:59")

import time

#  FUNGSI CHUNKING LIST
def _chunk_list(items, size=1000):
    """Memecah list teks menjadi potongan kecil agar aman untuk prompt."""
    for i in range(0, len(items), size):
        yield items[i:i + size]

def _call_gemini_retry(prompt, max_retries=8):
    delay = 1 
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="models/gemini-2.5-flash-lite",
                contents=prompt
            )

            try:
                return response.candidates[0].content.parts[0].text
            except:
                return response.text if hasattr(response, "text") else str(response)

        except Exception as e:
            err = str(e)

            if hasattr(e, "args") and len(e.args) > 0:
                if isinstance(e.args[0], dict):
                    err_dict = e.args[0].get("error", {})
                    err_code = err_dict.get("code", "")
                    err_msg  = err_dict.get("message", "")
                    if err_code == 503 or "overloaded" in err_msg:
                        pass 

            if ("503" in err) or ("overloaded" in err) or ("UNAVAILABLE" in err):
                if attempt < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 20)
                    continue
                else:
                    return f"⚠️ Gagal setelah {max_retries} percobaan: {err}"

            return f"⚠️ Error Gemini: {err}"

# INSET LEXICON
def _load_inset(pos_path="positive.tsv", neg_path="negative.tsv"):
    def _read(p):
        if not os.path.exists(p):
            return pd.DataFrame(columns=["word","weight"])
        df = pd.read_csv(p, sep="\t", header=None, names=["word","weight"],
                         dtype=str, encoding="utf-8-sig")
        df["word"] = df["word"].astype(str).str.strip().str.lower()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        return df.dropna()

    pos = _read(pos_path)
    neg = _read(neg_path)
    if pos.empty and neg.empty:
        raise FileNotFoundError("InSet tidak ditemukan.")

    lex = {}
    for _, r in pos.iterrows():
        lex[r["word"]] = lex.get(r["word"], 0.0) + float(r["weight"])
    for _, r in neg.iterrows():
        lex[r["word"]] = lex.get(r["word"], 0.0) + float(r["weight"])
    return lex

# UTIL PERIOD / QUARTER
def _quarter_label(p: pd.Period) -> str:
    return f"Q{p.quarter} ({p.start_time:%d %b %Y} – {p.end_time:%d %b %Y})"


def _pick_norm_text_col(df):
    for c in ["content_norm"]:
        if c in df.columns:
            return c
    objs = df.select_dtypes(include="object").columns
    return objs[0] if len(objs)>0 else df.columns[0]


def _extract_quarter(df):
    if "at_dt" not in df.columns:
        df["at_dt"] = pd.NaT

    if df["at_dt"].isna().all():
        df["_quarter"] = "UNKNOWN"
        return df

    df["_q"] = df["at_dt"].dt.to_period("Q-AUG")
    df["_quarter"] = df["_q"].map(_quarter_label)
    return df


def _group_quarter(df):
    if df.empty:
        return pd.DataFrame()

    # buang netral lebih dulu
    df2 = df[df["label"] != 0].copy()

    gp = (
        df2.groupby("_quarter", sort=False)
            .agg(
                total=("label", "count"),
                pos=("label", lambda s: int((s == 1).sum())),
                neg=("label", lambda s: int((s == -1).sum())),
            )
            .reset_index()
    )

    # menjaga urutan Q1–Q4
    if "_q" in df.columns:
        order = sorted(df["_q"].unique())
        order_map = {_quarter_label(q): i for i, q in enumerate(order)}
        gp["_ord"] = gp["_quarter"].map(order_map)
        gp = gp.sort_values("_ord").drop(columns="_ord")

    # hitung persentase
    gp["Positive Percentage"] = ((gp["pos"] / gp["total"]) * 100).round(2).astype(str) + " %"
    gp["Negative Percentage"] = ((gp["neg"] / gp["total"]) * 100).round(2).astype(str) + " %"

    # format angka ribuan pakai titik
    gp["Total"] = gp["total"].map(lambda x: f"{x:,}".replace(",", "."))
    gp["Positive"]   = gp["pos"].map(lambda x: f"{x:,}".replace(",", "."))
    gp["Negative"]   = gp["neg"].map(lambda x: f"{x:,}".replace(",", "."))

    return gp

# EVENT CLUES / TOP KEYWORDS PER KUARTAL
def _event_clues(df_q, k_sample=6):
    col = _pick_norm_text_col(df_q)
    s = df_q[col].astype(str)

    topic_keywords = [
        # Malware / Trojans
        r"malware|virus|trojan|phishing|berbahaya|bahaya|pencurian data|ambil data|data dicuri",

        # Tag tidak akurat
        r"tag.*(tidak akurat|salah|tidak sesuai|tidak muncul|kurang lengkap|error)"

        # # Keamanan aplikasi
        # r"aman|keamanan aplikasi|lindung|privacy|privasi|terpercaya|secure"

        # # Iklan Mengganggu
        # r"iklan.*(mengganggu|terlalu sering|banyak|ganggu|spam|menghalangi|pop ?up)",

        # # Akurasi Tag / kelengkapan informasi
        # r"tag.*(akurat|sesuai|benar|tepat|lengkap|update|muncul)"
    ]

    # Gabungkan pola regex
    pattern = "(" + "|".join(topic_keywords) + ")"

    # Cari yang match
    mask = s.str.contains(pattern, case=False, regex=True)

    cand = df_q.loc[mask].copy()
    if cand.empty:
        return []

    # Pilih komentar yang paling panjang (lebih informatif)
    cand["__len"] = s.loc[cand.index].map(lambda x: len(str(x).split()))
    cand = cand.sort_values("__len", ascending=False).drop_duplicates(subset=[col])

    out = cand.head(k_sample)

    # Format output
    items = []
    for _, r in out.iterrows():
        txt = str(r[col]).strip()

        dt = r.get("at_dt")
        if pd.notna(dt):
            items.append(f"- {dt:%d %b %Y}: {txt}")
        else:
            items.append(f"- {txt}")

    return items

# PREPROCESS + LABEL DATA BARU
def _preprocess_and_label_uploaded(df_raw):
    need = {"reviewId","at","content"}
    if not need.issubset(df_raw.columns):
        raise ValueError(f"Kolom wajib hilang: {need - set(df_raw.columns)}")

    df = df_raw.copy()
    df["at_dt"] = pd.to_datetime(df["at"], errors="coerce")

    # Normalisasi
    s = df["content"].fillna("").astype(str)
    s = s.str.replace(r"[^A-Za-zÀ-ÖØ-öø-ÿ\s]", " ", regex=True)
    s = s.str.replace(r"\s+"," ",regex=True).str.strip().str.lower()

    slang = load_slang_map("slangs.txt")
    df["content_norm"] = s.map(lambda t: normalize_with_slang(t, slang))

    df["tokens_join"] = df["content_norm"].map(lambda t: " ".join(t.split()))
    df = df[df["tokens_join"].str.strip()!=""]

    # Label InSet
    lex = _load_inset()

    def score_line(txt):
        sc = sum(lex.get(tok, 0.0) for tok in txt.split())
        return max(-5.0, min(5.0, sc))

    df["sent_score"] = df["tokens_join"].map(score_line)
    df["label"] = df["sent_score"].map(lambda s: (1 if s > 0 else (-1 if s < 0 else 0)))

    return df

# BASELINE DATA (DATA LAMA)
def _load_labelled_baseline():
    path = os.path.join(DATA_DIR, "label_only_app_source_getcontact.csv")

    if not path or not os.path.exists(path):
        st.info("Belum ada hasil label InSet.")
        return pd.DataFrame()

    df = load_csv(path).copy()
    if "tokens_join" not in df.columns or "label" not in df.columns:
        st.error("File label dasar perlu kolom 'tokens_join' dan 'label'.")
        return pd.DataFrame()

    if "at" in df.columns:
        df["at_dt"] = pd.to_datetime(df["at"], errors="coerce")
        df = df[(df["at_dt"]>=PERIOD_FROM)&(df["at_dt"]<=PERIOD_TO)]
    else:
        df["at_dt"] = pd.NaT

    df["label"] = df["label"].astype(int)

    if "content_norm" not in df.columns and "content" in df.columns:
        slang = load_slang_map("slangs.txt")
        s = df["content"].fillna("").astype(str).str.lower()
        s = s.str.replace(r"[^a-zà-öø-ÿ\s]"," ",regex=True).str.replace(r"\s+"," ",regex=True)
        df["content_norm"] = s.map(lambda t: normalize_with_slang(t, slang))

    return df

# CHART SETTINGS
def _axis_off():
    return alt.Axis(title=None, labels=False, ticks=False, grid=True)

def _legend_clean():
    return alt.Legend(title=None)

#  RENDER BLOK PER PERIODE (LAMA/BARU)
def _render_period_block(df_period, gp_period, tag):
    st.markdown(f"## Periode {tag}")

    if gp_period.empty:
        st.info(f"Tidak ada ringkasan untuk periode {tag}.")
        return

    # TABEL KUARTAL
    st.dataframe(
        gp_period[["_quarter","Total","Positive","Negative","Positive Percentage","Negative Percentage"]],
        use_container_width=True
    )

    # CHART
    gp_chart = gp_period.copy()

    # ubah persentase string “12.34 %” → angka float 12.34
    gp_chart["Positive_Pct_Num"] = gp_chart["Positive Percentage"].str.replace(" %","").astype(float)
    gp_chart["Negative_Pct_Num"] = gp_chart["Negative Percentage"].str.replace(" %","").astype(float)

    melt = gp_chart.melt(
        id_vars=["_quarter"],
        value_vars=["Positive_Pct_Num", "Negative_Pct_Num"],
        var_name="Kategori",
        value_name="Persen"
    )

    melt["Kategori"] = melt["Kategori"].map({
        "Positive_Pct_Num": "Positif",
        "Negative_Pct_Num": "Negatif",
    })

    st.altair_chart(
        alt.Chart(melt).mark_bar().encode(
            x=alt.X("_quarter:N", axis=_axis_off()),
            xOffset="Kategori:N",
            y=alt.Y("Persen:Q", axis=_axis_off()),
            color=alt.Color("Kategori:N", legend=_legend_clean()),
            tooltip=["_quarter","Kategori", alt.Tooltip("Persen:Q",format=".2f")]
        ).properties(height=280),
        use_container_width=True
    )

    # DETAIL PER KUARTAL (Q1–Q4)
    st.markdown(f"### Detail per Kuartal — Periode {tag}")

    if "_q" not in df_period.columns or df_period["_q"].isna().all():
        st.info("Tidak ada pembagian kuartal.")
        return

    for q in sorted(df_period["_q"].unique()):
        qlabel = _quarter_label(q)
        df_q = df_period[df_period["_q"] == q].copy()

        # buang komentar netral dari perhitungan
        df_q_non_neutral = df_q[df_q["label"] != 0]

        total_q = len(df_q_non_neutral)
        pos_q = int((df_q_non_neutral["label"] == 1).sum())
        neg_q = int((df_q_non_neutral["label"] == -1).sum())

        with st.expander(
            f"{qlabel} • {total_q:,} ulasan • Positif: {pos_q:,} • Negatif: {neg_q:,}",
            expanded=False
        ):
            col = _pick_norm_text_col(df_q)

            # kws = top_words(df_q[col].fillna(""), n=15)
            # if kws:
            #     st.write("**Top kata kunci:**")
            #     st.dataframe(
            #         pd.DataFrame(kws, columns=["kata","frekuensi"]),
            #         use_container_width=True
            #     )

            clues = _event_clues(df_q)
            if clues:
                st.write("**Cuplikan komentar:**")
                st.markdown("\n".join(clues))

    # KESIMPULAN OTOMATIS GEMINI
    st.markdown("### Kesimpulan Otomatis")
    st.write(_narrative_gemini(df_period, tag))

# HALAMAN UTAMA
def render():

    st.markdown("<h1 style='text-align:center;'>Analisis Tren Sentimen (Triwulan)</h1>", unsafe_allow_html=True)

    st.caption(
        "Jika tidak ada unggahan data baru, modul menampilkan dataset lama.<br>"
        "Jika diunggah Excel baru, sistem akan membersihkan → menormalisasi → "
        "men-token → melabeli (InSet) otomatis, lalu menampilkan perbandingan.",
        unsafe_allow_html=True
    )

    colA, colB = st.columns([5, 2])
    with colB:
        if st.button("🗑 Hapus Data Periode Baru"):
            if "tren_new_df" in st.session_state:
                del st.session_state["tren_new_df"]

            st.rerun() 
            
    st.markdown("---")

    st.markdown(
        "**Template Dataset (Excel) — Kolom wajib:** reviewId, at (YYYY-MM-DD HH:MM:SS), content"
    )

    tmpl = pd.DataFrame(
        [
            {"reviewId": "rid-0001", "at": "2025-09-01 10:35:12",
             "content": "Aplikasinya berguna, blokir spam efektif."},
            {"reviewId": "rid-0002", "at": "2025-09-03 08:02:01",
             "content": "Iklan terlalu sering dan tag tidak akurat."},
        ]
    )

    try:
        xbuf = io.BytesIO()
        tmpl.to_excel(xbuf, index=False, engine="openpyxl")
        st.download_button(
            "⬇️ Unduh Template (Excel)",
            xbuf.getvalue(),
            file_name="template_tren_sentimen.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception:
        st.caption("Tambahkan openpyxl di requirements agar unduhan Excel aktif.")

    up = st.file_uploader("Unggah dataset baru (Excel)", type=["xlsx"])
    df_new = pd.DataFrame()

    if up is not None:
        try:
            df_uploaded = pd.read_excel(up, engine="openpyxl")
            df_new = _preprocess_and_label_uploaded(df_uploaded)
            st.success(f"Dataset baru diproses otomatis: {len(df_new):,} ulasan.")
            st.session_state["tren_new_df"] = df_new
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

    if df_new.empty and "tren_new_df" in st.session_state:
        df_new = st.session_state["tren_new_df"]

    # LOAD DATA LAMA + RENDER
    df_old = _load_labelled_baseline()
    if df_old.empty:
        return

    df_old = _extract_quarter(df_old.copy())
    gp_old = _group_quarter(df_old)

    if not df_new.empty:
        df_new = _extract_quarter(df_new.copy())
        gp_new = _group_quarter(df_new)

        col1, col2 = st.columns(2)
        with col1:
            _render_period_block(df_old, gp_old, tag="Lama")
        with col2:
            _render_period_block(df_new, gp_new, tag="Baru")
    else:
        _render_period_block(df_old, gp_old, tag="Lama")

#   RINGKASAN GEMINI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache_summaries")
os.makedirs(CACHE_DIR, exist_ok=True)


def _narrative_gemini(df: pd.DataFrame, tag: str) -> str:
    """
    Untuk 'Lama' → SELALU ambil dari file cache manual.
    Untuk 'Baru' → pakai Gemini + auto-cache seperti biasa.
    """
    import hashlib
    import json

    # --- PERIODE LAMA: gunakan file kesimpulan yang sudah ada ---
    if tag.lower() == "lama":
        manual_path = os.path.join(
            DATA_DIR, "..", ".cache_summaries",
            "Lama_406b07680d87552cea01fbb5ea845ae9.txt"
        )

        manual_path = os.path.abspath(manual_path)

        if os.path.exists(manual_path):
            with open(manual_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return "⚠️ File kesimpulan periode lama tidak ditemukan."

    # --- PERIODE BARU: proses seperti biasa ---
    if df.empty:
        return f"Periode {tag} tidak memiliki data."

    text_col = _pick_norm_text_col(df)
    texts = df[text_col].fillna("").astype(str).tolist()
    if len(texts) > 20000:
        texts = texts[:20000]

    h = hashlib.md5(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()

    safe_tag = str(tag).replace(" ", "_")
    cache_key = f"{safe_tag}_{h}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.txt")

    # gunakan cache otomatis jika ada
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    # buat ringkasan baru via Gemini
    chunks = list(_chunk_list(texts, size=1000))[:20]
    partial_summaries = []

    for chunk in chunks:
        joined = "\n".join(chunk)[:120_000]

        prompt = f"""
Kamu adalah analis data profesional.

Buat ringkasan sentimen dari ulasan pengguna aplikasi Getcontact untuk periode **{tag}**.
Gunakan format yang FORMAL, SISTEMATIS, dan tidak mengada-ngada.

======================================================================
🟢 **1. SENTIMEN POSITIF**
Kelompokkan ulasan positif ke dalam subkategori berikut:

A. **Keamanan Aplikasi**
   - Apresiasi terhadap keamanan, kenyamanan, atau kepercayaan pengguna.

B. **Fitur Premium / Langganan**
   - Hal positif tentang fitur premium, manfaat berlangganan, atau value for money.

C. **Akurasi Tagar / Informasi**
   - Pujian mengenai akurasi tag, kelengkapan informasi, atau hasil pencarian.

D. **Pengalaman Penggunaan / Antarmuka**
   - Kemudahan digunakan, tampilan aplikasi, navigasi, atau pengalaman pengguna yang baik.

E. **Performa Aplikasi / Stabilitas**
   - Hal positif mengenai kecepatan, login lancar, tidak error, stabil, atau update yang memperbaiki fitur.

F. **Hal Positif Lainnya**
   - Semua pujian atau apresiasi yang tidak termasuk kategori di atas.

======================================================================
🔴 **2. SENTIMEN NEGATIF**
Kelompokkan ulasan negatif ke dalam subkategori berikut:

A. **Malware / Trojan / Virus / Keamanan Data**
   - Tuduhan bahwa aplikasi berbahaya, mencurigakan, mengambil data, atau terkait risiko keamanan. (data ada yang membahas ini)

B. **Fitur Premium / Langganan Berbayar**
   - Keluhan mengenai biaya premium, fitur dibatasi, premium tidak berfungsi, tagar terkunci, atau penagihan otomatis.

C. **Tagar Tidak Akurat / Tidak Muncul**
   - Tag tidak sesuai, tidak lengkap, tidak update, atau hasil yang salah. (data ada yang membahas ini)

D. **Iklan Mengganggu**
   - Keluhan tentang iklan yang terlalu sering, tidak sesuai, atau mengganggu. (data ada yang membahas ini)

E. **Error Update / Masalah Teknis**
   - Kendala login, OTP tidak masuk, aplikasi tidak bisa dibuka, crash, bug, atau gangguan teknis lain.

F. **Keluhan Umum Lainnya**
   - Keluhan signifikan di luar kategori di atas.

======================================================================
Gunakan bahasa formal, ringkas, namun menyeluruh.
Jangan mengarang hal yang tidak ada di data.

Berikut ulasan pengguna:
--------------------------------------------------
{joined}
--------------------------------------------------
"""

        piece = _call_gemini_retry(prompt)
        if piece is None:
            piece = "Ringkasan tidak tersedia (API error)."

        partial_summaries.append(str(piece))

    final_prompt = f"""
Gabungkan seluruh ringkasan berikut menjadi satu ringkasan akhir periode **{tag}**:

{chr(10).join(partial_summaries)}
"""

    final_summary = _call_gemini_retry(final_prompt)
    if final_summary is None:
        final_summary = "Ringkasan final tidak dapat dibuat (API error)."

    # simpan cache otomatis
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(final_summary)

    return final_summary

if __name__ == "__main__":
    render()
