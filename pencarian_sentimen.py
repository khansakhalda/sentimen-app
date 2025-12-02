import os, io, math
import numpy as np
import pandas as pd
import streamlit as st
from helpers import (
    DATA_DIR, DEFAULT_TABLE_PAGE_SIZE, load_csv, latest,
    multi_match_mask
)

BADGE_CSS = """
<style>
.label-pos {background:#16a34a; color:white; padding:2px 8px; border-radius:999px; font-size:12px}
.label-neg {background:#dc2626; color:white; padding:2px 8px; border-radius:999px; font-size:12px}
.smallcap {color:#64748b; font-size:12px}
.kpill{display:inline-block;margin:0 .25rem .25rem 0;padding:.15rem .5rem;border-radius:999px;background:#eef2ff;border:1px solid #c7d2fe;font-size:12px}
.tip{background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;padding:.6rem .8rem;margin:.5rem 0}
</style>
"""

def _ensure_label_text(df):
    if "label_text" in df.columns:
        return df

    if "label" in df.columns:

        def _labtxt(x):
            s = str(x).strip()

            if s == "1":
                return "positif"
            elif s == "-1":
                return "negatif"
            elif s == "0":
                return "netral"
            return "netral"

        df = df.copy()
        df["label_text"] = df["label"].map(_labtxt)
        return df

    return df

def render():
    st.markdown("<h1 style='text-align:center;'>Pencarian Sentimen</h1>", unsafe_allow_html=True)
    st.write("Cari sentimen berdasarkan kata/frasa pada hasil labeling.")
    st.markdown(BADGE_CSS, unsafe_allow_html=True)

    # CEK DATA LABEL
    label_path = latest(os.path.join(DATA_DIR, "label_only_app_source_getcontact.csv"))
    if not label_path:
        st.info("Belum ada hasil label InSet.")
        return

    df = load_csv(label_path)
    df = _ensure_label_text(df)

    # INIT SESSION STATE
    st.session_state.setdefault("search_query", "")
    st.session_state.setdefault("search_filter", "Semua")
    st.session_state.setdefault("search_page_size", DEFAULT_TABLE_PAGE_SIZE)
    st.session_state.setdefault("sa_page", 1)

    # FORM QUERY
    with st.form("search_form", clear_on_submit=False):
        query = st.text_input(
            "Masukkan kata (pisahkan dengan koma)",
            st.session_state.search_query,
            placeholder="Contoh: tag, premium, iklan; atau: blokir nomor"
        )
        submitted = st.form_submit_button("Terapkan")

    if submitted:
        st.session_state.search_query = query
        st.session_state.sa_page = 1

    # SELECTBOX DI LUAR FORM
    sent_filter = st.selectbox(
        "Filter sentimen",
        ["Semua", "Positif", "Negatif"],
        index=["Semua", "Positif", "Negatif"].index(st.session_state.search_filter)
    )
    st.session_state.search_filter = sent_filter

    page_size = st.selectbox(
        "Baris per halaman",
        [10, 20, 30, 50, 100],
        index=[10,20,30,50,100].index(st.session_state.search_page_size)
    )
    st.session_state.search_page_size = page_size

    # INTERPRETASI INPUT
    col = "tokens_join" if "tokens_join" in df.columns else df.columns[0]
    terms = [t.strip() for t in (st.session_state.search_query or "").split(",") if t.strip()]
    WHOLE_WORD = True

    # tampilkan info pencarian
    if terms:
        chips = " ".join([f"<span class='kpill'>{t}</span>" for t in terms])
        logic = "1 kata → mencari ulasan yang mengandung kata tersebut." \
                if len(terms) == 1 else f"{len(terms)} kata → semua kata harus muncul."
        st.markdown(
            f"<div class='tip'><b>Kata kunci:</b> {chips}<br>"
            f"<b>Aturan:</b> {logic}<br>"
            f"<b>Metode:</b> pencocokan kata utuh.</div>",
            unsafe_allow_html=True
        )
    else:
        with st.expander("Cara pakai & contoh"):
            st.markdown(
                "- Ketik satu atau beberapa kata dipisahkan koma.\n"
                "- Misal: `tagar, premium` → mencari ulasan yang memuat kedua kata."
            )

    # FILTER DATA BERDASARKAN QUERY
    if terms:
        mode_all = len(terms) > 1
        mask = multi_match_mask(df[col], terms, WHOLE_WORD, mode_all)
        sub = df.loc[mask].copy().reset_index(drop=True)
    else:
        sub = df.copy()

    # CEK KOLOM LABEL
    if "label_text" in sub.columns:
        label_col = "label_text"
    elif "label" in sub.columns:
        label_col = "label"
    else:
        label_col = None

    # fungsi penilai
    def _is_pos(v): 
        return str(v).strip().lower() in {"1", "positif", "positive"}

    def _is_neg(v): 
        return str(v).strip().lower() in {"-1", "negatif", "negative"}

    def _is_neu(v): 
        return str(v).strip().lower() in {"0", "netral", "neutral"}

    # =======================================
    # 🔥 HAPUS SEMUA KOMENTAR NETRAL
    # =======================================
    if label_col:
        sub = sub[~sub[label_col].map(_is_neu)]

    # filter sentimen (positif / negatif)
    if label_col and sent_filter != "Semua":
        sub = sub[sub[label_col].map(_is_pos if sent_filter == "Positif" else _is_neg)]

    # JIKA TIDAK ADA HASIL → TAMPILKAN TABEL KOSONG
    if sub.empty:
        st.warning("Tidak ada hasil ditemukan untuk pencarian ini.")

        if label_col:
            empty_df = pd.DataFrame(columns=["No", "Ulasan", "Kategori"])
            st.markdown(empty_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            empty_df = pd.DataFrame(columns=["No", "Ulasan"])
            st.dataframe(empty_df, use_container_width=True)

        return

    # RINGKASAN
    if label_col:
        pos_n = int(sub[label_col].map(_is_pos).sum())
        neg_n = int(sub[label_col].map(_is_neg).sum())
        st.caption(
            f"<span class='smallcap'>Positif: {pos_n:,} • Negatif: {neg_n:,} • Total: {len(sub):,}</span>",
            unsafe_allow_html=True
        )
    else:
        st.caption(
            f"<span class='smallcap'>Total: {len(sub):,} • (kolom sentimen tidak ditemukan)</span>",
            unsafe_allow_html=True
        )

    # PAGINATION
    total = len(sub)
    total_pages = max(1, math.ceil(total / page_size))
    current_page = min(st.session_state.sa_page, total_pages)

    start = (current_page - 1) * page_size
    end = start + page_size

    view = sub.iloc[start:end].copy()
    view["No"] = np.arange(start + 1, start + 1 + len(view))
    view["Ulasan"] = view[col]

    # badge kategori
    if label_col:
        def labfmt(x):
            if _is_pos(x): return "<span class='label-pos'>positif</span>"
            if _is_neg(x): return "<span class='label-neg'>negatif</span>"
            return str(x)
        view["Kategori"] = view[label_col].map(labfmt)

    # tampilkan tabel
    if label_col:
        st.markdown(
            view[["No", "Ulasan", "Kategori"]].to_html(escape=False, index=False),
            unsafe_allow_html=True
        )
    else:
        st.dataframe(view[["No", "Ulasan"]], use_container_width=True)

    # NAVIGASI
    disable_prev = current_page <= 1
    disable_next = current_page >= total_pages

    left, mid, right = st.columns([1,2,1])

    with left:
        st.caption(f"Total hasil: {total:,}")

    with mid:
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("⏮️", disabled=disable_prev):
            st.session_state.sa_page = 1
            st.rerun()
        if c2.button("⬅️", disabled=disable_prev):
            st.session_state.sa_page = current_page - 1
            st.rerun()
        if c3.button("➡️", disabled=disable_next):
            st.session_state.sa_page = current_page + 1
            st.rerun()
        if c4.button("⏭️", disabled=disable_next):
            st.session_state.sa_page = total_pages
            st.rerun()

    with right:
        st.caption(f"Halaman {current_page} / {total_pages}")

# # Unduh hasil
# csv_buf = io.StringIO()
# to_save = sub.copy()
# if "label_text" in to_save.columns:
#     to_save["label_text"] = to_save["label_text"].astype(str)
# to_save.to_csv(csv_buf, index=False, encoding="utf-8-sig")
# st.download_button("⬇️ Unduh hasil (CSV)", csv_buf.getvalue(),
#                    file_name="hasil_pencarian_sentimen.csv", mime="text/csv")
