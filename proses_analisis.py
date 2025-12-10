import os, time
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import base64

# Pastikan backend non-interactive dipakai agar figure tetap valid di Streamlit loop
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helpers import (
    DATA_DIR, BASE_SIMPLE,
    load_csv, latest, top_words, build_wc_and_freq_after, build_wc_and_freq_before,
    offer_download, show_images_side_by_side, btn, has_param,
    auto_insight_from_raw, load_slang_map, normalize_with_slang
)

import pipeline as mod
mod_path = mod.__file__

import pickle

STATE_FILE = "app_state.pkl"

SAFE_KEYS = {
    "res_collect",
    "res_clean",
    "res_casefold",
    "res_normalize",
    "res_tokenize",
    "res_label",
    "res_tfidf_feat",
    "res_split_multi",
    "svm_linear_multi",
    "svm_rbf_multi",
    "norm_stats",
    "tok_stats"
}

def save_state():
    safe_data = {k: v for k, v in st.session_state.items() if k in SAFE_KEYS}
    with open("app_state.pkl", "wb") as f:
        pickle.dump(safe_data, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            data = pickle.load(f)

            BLOCKED = {"page", "pkg"}

            for k, v in data.items():
                if k in SAFE_KEYS and k not in BLOCKED:
                    if k not in st.session_state:
                        st.session_state[k] = v

def info_text(text: str):
    """Tampilkan teks deskripsi biasa (hitam, tanpa kotak biru)."""
    st.markdown(
        f"<p style='color:black; font-size:16px; margin:0 0 0.4rem 0;'>{text}</p>",
        unsafe_allow_html=True,
    )

def render():
    # 🔄 Reset semua data — dipindah ke dalam render()
    # if st.button("🔄 Reset Semua Data"):
    #     if os.path.exists(STATE_FILE):
    #         os.remove(STATE_FILE)
    #     st.session_state.clear()
    #     st.rerun()

    # ================= HEADER =================
    st.markdown(
        """
        <h1 style='text-align:center; font-weight:700;'>
        Aplikasi Analisis Sentimen Ulasan Google Play Store
        </h1>
        <p style='text-align:justify; font-size:16px;'>
        Aplikasi ini mengintegrasikan <b>InSet Lexicon</b> untuk pelabelan otomatis dan 
        <b>Support Vector Machine (SVM)</b> dengan kernel <b>Linear</b> dan <b>RBF</b>.
        </p>
        """,
        unsafe_allow_html=True
    )

    # === Package ID (dikunci ke Getcontact) ===
    GETCONTACT_PKG = "app.source.getcontact"
    st.session_state["pkg"] = GETCONTACT_PKG
    st.text_input("Package ID", value=GETCONTACT_PKG, disabled=True)

    # ================= TAHAP 1 =================
    st.subheader("Tahap 1 — Pengumpulan Data")
    info_text(
        "Tahap ini mengambil data ulasan dari Google Play Store agar dapat digunakan "
        "sebagai dasar analisis selanjutnya."
    )

    # 🔥 TOMBOL PENGUMPULAN DATA
    # if btn("📥 Mulai Pengumpulan Data", key="t1_collect"):
    #     pkg = (st.session_state.get("pkg") or "").strip()
    #     if not pkg:
    #         st.warning("Isi dulu Package ID")
    #         st.stop()

    #     fn = getattr(mod, "stage1_collect", None)
    #     if not fn:
    #         st.error("Fungsi stage1_collect tidak ditemukan.")
    #         st.stop()

    #     live = st.empty()
    #     t0 = time.time()

    #     def cb(n):
    #         dt = time.time() - t0
    #         spd = n / dt if dt > 0 else 0.0
    #         live.info(f"Mengunduh ulasan: {n:,} • {spd:.2f} ulasan/dtk • {dt:,.0f} dtk")

    #     kwargs = dict(
    #         app_input=pkg,
    #         max_reviews=0,
    #         page_size=200,
    #         lang="id",
    #         country="id",
    #         out_dir=DATA_DIR,
    #     )
    #     if has_param(fn, "progress_callback"):
    #         kwargs["progress_callback"] = cb

    #     with st.spinner("Mengambil ulasan dari Google Play..."):
    #         ret = fn(**kwargs)

    #     out_csv, wc_png, freq_png = (ret if isinstance(ret, tuple) else (ret, None, None))
    #     elapsed = time.time() - t0

    #     live.empty()
    #     st.session_state["res_collect"] = {
    #         "out_csv": out_csv,
    #         "wc_png": wc_png,
    #         "freq_png": freq_png,
    #         "elapsed": elapsed,
    #     }
    #     save_state()
    #     st.rerun()

    if st.session_state.get("res_collect"):
        rc = st.session_state["res_collect"]
        raw_path = "data/raw_app_source_getcontact.csv"

        df_raw = load_csv(raw_path)

        st.caption(
            f"CSV tersimpan: {os.path.basename(raw_path)} • {rc['elapsed']:,.1f} dtk"
        )

        colA, colB = st.columns(2)
        total = len(df_raw)
        at_min = pd.to_datetime(df_raw['at'], errors='coerce').min()
        at_max = pd.to_datetime(df_raw['at'], errors='coerce').max()
        with colA:
            st.metric("Total ulasan", f"{total:,}")
        with colB:
            st.metric("Rentang tanggal", f"{at_min:%d %b %y} – {at_max:%d %b %y}")

        st.markdown("**📄 Preview Data Mentah**")
        st.dataframe(df_raw.head(10), use_container_width=True)
        from helpers import make_download_button
        st.markdown(
            make_download_button(
                raw_path,
                "Unduh Data Mentah",
                "raw_app_source_getcontact.csv"
            ),
            unsafe_allow_html=True
        )
        # --- FIX DOWNLOAD BUTTON ---
        # with open(raw_path, "rb") as f:
        #     st.download_button(
        #         label="⬇️ Unduh Data Mentah",
        #         data=f.read(),
        #         file_name="raw_app_source_getcontact.csv",
        #         mime="text/csv"
        #     )
        # offer_download(raw_path, "⬇️ Unduh Data Mentah")

        # # ===== Scatter Plot Frekuensi Kata (Top-10) =====
        # st.markdown("**📌 Scatter Plot Frekuensi Kata (Top-10, sebelum preprocessing)**")

        # # ambil top-10 saja
        # df_scatter = pd.DataFrame(top_words_pre, columns=["kata", "frekuensi"]).head(10)

        # scatter_chart = (
        #     alt.Chart(df_scatter)
        #     .mark_circle(size=250, color="#1f77b4")  # titik lebih besar & jelas
        #     .encode(
        #         x=alt.X("frekuensi:Q", title="Frekuensi"),
        #         y=alt.Y("kata:N", sort='-x', title="Kata"),
        #         tooltip=["kata", "frekuensi"]
        #     )
        #     .properties(height=330)
        # )

        # st.altair_chart(scatter_chart, use_container_width=True)

        # wc_before = os.path.join(DATA_DIR, "wc_before.png")
        freq_before = os.path.join(DATA_DIR, "freq_before.png")

        build_wc_and_freq_before(raw_path, None, freq_before)

        imgs, caps = [], []

        # if os.path.exists(wc_before):
        #     imgs.append(wc_before)
        #     caps.append("WordCloud (sebelum preprocessing)")

        if os.path.exists(freq_before):
            imgs.append(freq_before)
            caps.append("Top-15 Frekuensi Kata (sebelum preprocessing)")

        show_images_side_by_side(imgs, caps, width=500)

        st.markdown(
            "<h4 style='font-size:20px; margin-top:25px;'>🧠 Auto-Insight</h4>",
            unsafe_allow_html=True
        )

        for tip in auto_insight_from_raw(df_raw):
            info_text("• " + tip)

    st.divider()

    # ================= TAHAP 2 =================
    st.subheader("Tahap 2 — Text Preprocessing")

    # Cleaning
    st.markdown("#### 🧹 Data Cleaning")
    info_text(
        "Tahap ini membersihkan teks dengan menghapus karakter yang tidak diperlukan "
        "seperti emoji, angka, tanda baca, dan URL agar data lebih bersih."
    )

    # if btn("1️⃣ Jalankan Data Cleaning", key="t2_clean"):
    #     if not st.session_state.get("res_collect"):
    #         st.warning("Jalankan Tahap 1 terlebih dahulu.")
    #         st.stop()

    #     src = st.session_state["res_collect"]["out_csv"]
    #     ##out = os.path.join(DATA_DIR, f"clean_{BASE_SIMPLE}.csv")
    #     out = os.path.join(DATA_DIR, "clean.csv")

    #     with st.spinner("Membersihkan karakter non-huruf..."):
    #         df = load_csv(src).copy()
    #         df["content_clean"] = df["content"].fillna("").astype(str).str.replace(
    #             r"[^A-Za-z\s]", " ", regex=True
    #         )
    #         df.to_csv(out, index=False, encoding="utf-8-sig")

    #     st.session_state["res_clean"] = out
    #     st.success(f"✅ Selesai: {os.path.basename(out)}")
    #     save_state()
    #     st.rerun()

    # if st.session_state.get("res_clean"):
    #     p = st.session_state["res_clean"]
    #     st.caption(f"Cleaning terakhir: {os.path.basename(p)}")
    #     st.markdown("**📄 Preview Data Cleaning**")
    #     st.dataframe(load_csv(p).head(10), use_container_width=True)
    #     offer_download(p, "⬇️ Unduh Data Cleaning")

    # st.markdown("---")

    clean_path = "data/clean.csv"

    if os.path.exists(clean_path):
        df_clean = load_csv(clean_path)
        st.caption(f"Cleaning terakhir: {os.path.basename(clean_path)}")
        st.markdown("**📄 Preview Data Cleaning**")
        st.dataframe(df_clean.head(10), use_container_width=True)

        st.markdown(
            make_download_button(
                clean_path,
                "Unduh Data Cleaning",
                "clean.csv"
            ),
            unsafe_allow_html=True
        )
    else:
        st.warning("File clean.csv tidak ditemukan.")

    # Case folding
    st.markdown("#### 🔡 Case Folding")
    info_text(
        "Tahap ini mengubah seluruh teks menjadi huruf kecil agar penulisan kata menjadi konsisten."
    )

    # if btn("2️⃣ Jalankan Case Folding", key="t2_case"):
    #     if not st.session_state.get("res_clean"):
    #         st.warning("Jalankan Data Cleaning dulu.")
    #         st.stop()

    #     src = st.session_state["res_clean"]
    #     ##out = os.path.join(DATA_DIR, f"casefold_{BASE_SIMPLE}.csv")
    #     out = os.path.join(DATA_DIR, "casefold.csv")

    #     with st.spinner("Mengubah menjadi huruf kecil..."):
    #         df = load_csv(src).copy()
    #         base_col = "content_clean" if "content_clean" in df.columns else "content"
    #         df["content_casefold"] = df[base_col].astype(str).str.lower()
    #         df.to_csv(out, index=False, encoding="utf-8-sig")

    #     st.session_state["res_casefold"] = out
    #     save_state()
    #     st.success(f"✅ Selesai: {os.path.basename(out)}")
    #     st.rerun()

    casefold_path = "data/casefold.csv"

    if st.session_state.get("res_casefold"):
        # p = st.session_state["res_casefold"]
        p = "data/casefold.csv"
        st.caption(f"Casefold terakhir: {os.path.basename(p)}")
        st.markdown("**📄 Preview Data Case Folding**")
        st.dataframe(load_csv(p).head(10), use_container_width=True)
        st.markdown(
            make_download_button(
                casefold_path,
                "Unduh Case Folding",
                "casefold.csv"
            ),
            unsafe_allow_html=True
        )
    else:
        st.warning("File casefold.csv tidak ditemukan.")

    # Normalization
    st.markdown("#### 🧾 Text Normalization")
    info_text(
        "Tahap ini merapikan teks dan mengganti kata tidak baku atau kata gaul menjadi "
        "bentuk yang sesuai agar kalimat lebih standar."
    )

    norm_path = "data/norm.csv"

    # if btn("3️⃣ Jalankan Text Normalization", key="t2_norm"):
    #     if not st.session_state.get("res_casefold"):
    #         st.warning("Jalankan Case Folding dulu.")
    #         st.stop()

    #     src = st.session_state["res_casefold"]
    #     ##out = os.path.join(DATA_DIR, f"norm_{BASE_SIMPLE}.csv")
    #     out = os.path.join(DATA_DIR, "norm.csv")

    #     with st.spinner("Normalisasi teks..."):
    #         df = load_csv(src).copy()
    #         base_col = "content_casefold" if "content_casefold" in df.columns else "content_clean"
    #         s = (
    #             df[base_col]
    #             .fillna("")
    #             .astype(str)
    #             .str.replace(r"\s+", " ", regex=True)
    #             .str.strip()
    #             .str.lower()
    #         )

    #         slang = load_slang_map("slangs.txt")
    #         df["content_norm"] = s.map(lambda t: normalize_with_slang(t, slang))

    #         before = len(df)
    #         df = df[df["content_norm"].astype(str).str.strip() != ""].copy()
    #         removed = before - len(df)

    #         df.to_csv(out, index=False, encoding="utf-8-sig")

    #     st.session_state["res_normalize"] = out
    #     save_state()
    #     st.session_state["norm_stats"] = {"before": before, "after": len(df), "removed": removed}
    #     st.success(f"✅ Selesai: {os.path.basename(out)}")
    #     st.rerun()

    if st.session_state.get("res_normalize"):
        p = "data/norm.csv"
        df_norm = load_csv(p)
        stats = st.session_state.get(
            "norm_stats",
            {"before": None, "after": len(df_norm), "removed": None},
        )

        st.caption(f"Normalisasi terakhir: {os.path.basename(p)}")
        st.success(
            "Total ulasan setelah normalisasi: "
            f"{len(df_norm):,}"
            + (f" • dihapus: {stats.get('removed'):,}" if stats.get("removed") is not None else "")
        )
        st.markdown("**📄 Preview Data Text Normalization**")
        st.dataframe(df_norm.head(10), use_container_width=True)
        st.markdown(
            make_download_button(
                norm_path,
                "Unduh Text Normalization",
                "norm.csv"
            ),
            unsafe_allow_html=True
        )

    else:
        st.warning("File norm.csv tidak ditemukan.")

    # Tokenizing
    st.markdown("#### ✂️ Words Tokenizing")
    info_text(
        "Tahap ini memecah setiap kalimat menjadi daftar kata agar analisis dapat "
        "dilakukan pada level token."
    )

    token_path = "data/token.csv"
    # if btn("4️⃣ Jalankan Words Tokenizing", key="t2_tok"):
    #     if not st.session_state.get("res_normalize"):
    #         st.warning("Jalankan Text Normalization dulu.")
    #         st.stop()

    #     src = st.session_state["res_normalize"]
    #     ##out = os.path.join(DATA_DIR, f"token_{BASE_SIMPLE}.csv")
    #     out = os.path.join(DATA_DIR, "token.csv")

    #     with st.spinner("Tokenizing kata..."):
    #         df = load_csv(src).copy()
    #         base_col = "content_norm" if "content_norm" in df.columns else "content_casefold"

    #         df["tokens"] = df[base_col].fillna("").astype(str).map(lambda s: s.split())
    #         df["tokens_join"] = df["tokens"].map(lambda xs: " ".join(xs).strip())

    #         before = len(df)
    #         df = df[df["tokens_join"].astype(str).str.strip() != ""].copy()
    #         removed = before - len(df)

    #         df.to_csv(out, index=False, encoding="utf-8-sig")

    #     st.session_state["res_tokenize"] = out
    #     save_state()
    #     st.session_state["tok_stats"] = {"before": before, "after": len(df), "removed": removed}
    #     st.success(f"✅ Selesai: {os.path.basename(out)}")
    #     st.rerun()

    if st.session_state.get("res_tokenize"):
        p = "data/token.csv"
        df_tok_full = load_csv(p)
        stats = st.session_state.get(
            "tok_stats",
            {"after": len(df_tok_full), "removed": None},
        )

        st.caption(f"Tokenizing terakhir: {os.path.basename(p)}")
        # st.success(
        #     "Total ulasan setelah tokenisasi: "
        #     f"{len(df_tok_full):,}"
        #     + (f" • dihapus: {stats.get('removed'):,}" if stats.get("removed") is not None else "")
        # )
        st.markdown("**📄 Preview Data Words Tokenizing**")
        st.dataframe(df_tok_full.head(10), use_container_width=True)
        st.markdown(
            make_download_button(
                token_path,
                "Unduh Words Tokenizing",
                "token.csv"
            ),
            unsafe_allow_html=True
        )

        texts = df_tok_full["tokens_join"].dropna().astype(str).tolist()
        # wc_after = os.path.join(DATA_DIR, "wc_after_preprocess.png")
        freq_after = os.path.join(DATA_DIR, "freq_after_preprocess.png")

        build_wc_and_freq_after(texts, None, freq_after)

        imgs, caps = [], []
        # if os.path.exists(wc_after):
        #     imgs.append(wc_after)
        #     caps.append("WordCloud (setelah preprocessing)")
        if os.path.exists(freq_after):
            imgs.append(freq_after)
            caps.append("Top-15 Frekuensi Kata (setelah preprocessing)")

        if imgs:
            show_images_side_by_side(imgs, caps, width=520)

    else:
        st.warning("File token.csv tidak ditemukan.")

        # top_after = top_words(df_tok_full["tokens_join"].fillna(""), n=20)
        # if top_after:
        #     st.markdown("**📈 Top kata (setelah preprocessing)**")
        #     st.dataframe(
        #         pd.DataFrame(top_after, columns=["kata", "frekuensi"]),
        #         use_container_width=True,
        #         height=260,
        #     )

    st.divider()

    # # ================= TAHAP 3 =================
    # st.subheader("Tahap 3 — Data Labelling menggunakan InSet Lexicon")
    # info_text(
    #     "Tahap ini memberikan label sentimen positif atau negatif secara otomatis "
    #     "menggunakan kamus InSet."
    # )

    # label_path = "data/label_only_app_source_getcontact.csv"

    # #Tombol untuk menjalankan labelling
    # if btn("Jalankan Labelling (InSet)", key="t3_label_inset"):
    #     fn = getattr(mod, "stage6_label_inset", None)
    #     if not fn:
    #         st.error("Fungsi stage6_label_inset tidak ditemukan.")
    #     else:
    #         with st.spinner("Menjalankan Labeling InSet..."):
    #             out = fn()
    #         st.session_state["res_label"] = out
    #         st.success(f"✅ Selesai: {os.path.basename(out)}")
    #         save_state()
    #         st.rerun()

    # Jika hasil labelling sudah tersedia
    # ================= TAHAP 3 =================
    st.subheader("Tahap 3 — Data Labelling menggunakan InSet Lexicon")
    info_text(
        "Tahap ini memberikan label sentimen positif atau negatif secara otomatis "
        "menggunakan kamus InSet."
    )

    label_path = "data/label_only_app_source_getcontact.csv"

    # ==== TAMPILKAN HASIL LABELLING =====
    if st.session_state.get("res_label"):

        p = "data/label_only_app_source_getcontact.csv"
        st.caption(f"Label terbaru: {os.path.basename(p)}")

        df_lab = load_csv(p)

        # Hitung jumlah kategori
        total_docs = len(df_lab)
        pos_n = int((df_lab["label"] == 1).sum())
        neg_n = int((df_lab["label"] == -1).sum())
        neu_n = int((df_lab["label"] == 0).sum())

        st.success(
            f"Positif: {pos_n:,} | Negatif: {neg_n:,} | Netral: {neu_n:,} | Total: {total_docs:,}"
        )

        # Preview data
        df_show = df_lab.copy()
        df_show.insert(0, "No", np.arange(1, len(df_show) + 1))

        st.markdown("**📄 Preview Data Labelling**")
        st.dataframe(df_show.head(10), use_container_width=True)

        st.markdown(
            make_download_button(
                label_path,
                "Unduh Label (InSet)",
                os.path.basename(label_path)
            ),
            unsafe_allow_html=True
        )

    else:
        st.warning("File label hasil InSet tidak ditemukan.")
        st.stop()

    # ================= VISUALISASI SELALU TAMPIL =================
    st.markdown(
        "<h4 style='font-size:20px; margin-top:25px;'>📊 Distribusi Label</h4>",
        unsafe_allow_html=True
    )

    # Ambil kategori lengkap
    vc = (
        df_lab["label"]
        .map({1: "Positif", -1: "Negatif", 0: "Netral"})
        .value_counts()
        .reset_index()
    )
    vc.columns = ["Kategori", "Jumlah"]
    vc["JumlahLabel"] = vc["Jumlah"].apply(lambda x: f"{x:,}".replace(",", "."))

    # ==== Paksa urutan Kategori ====
    order_kat = ["Negatif", "Positif", "Netral"]
    vc["Kategori"] = pd.Categorical(vc["Kategori"], categories=order_kat, ordered=True)

    # ==== Warna ====
    custom_colors = alt.Scale(
        domain=order_kat,
        range=["#ff9bb8", "#8ab6ff", "#d9d9d9"]  # Negatif, Positif, Netral
    )

    bars = (
        alt.Chart(vc)
        .mark_bar()
        .encode(
            x=alt.X("Kategori:N", sort=order_kat, title="Kategori"),
            y=alt.Y("Jumlah:Q", title="Jumlah", axis=alt.Axis(format=",.0f")),
            color=alt.Color("Kategori:N", scale=custom_colors, title="Kategori"),
            tooltip=["Kategori", "JumlahLabel"]
        )
    )

    labels = (
        alt.Chart(vc)
        .mark_text(
            dy=-5,
            fontSize=12,
            color="black"
        )
        .encode(
            x=alt.X("Kategori:N", sort=order_kat),
            y="Jumlah:Q",
            text="JumlahLabel:N"
        )
    )

    st.altair_chart(bars + labels, use_container_width=True)

    # ================= PIE CHART PERSENTASE =================
    st.markdown(
        "<h4 style='font-size:22px; margin-top:25px;'>📌 Persentase Sentimen</h4>",
        unsafe_allow_html=True
    )

    # Hitung kategori
    vc = (
        df_lab["label"]
        .map({1: "Positif", -1: "Negatif", 0: "Netral"})
        .value_counts()
        .reset_index()
    )
    vc.columns = ["Kategori", "Jumlah"]

    # Persentase + format koma
    vc["Persen"] = (vc["Jumlah"] / vc["Jumlah"].sum() * 100).round(1)
    vc["PersenStr"] = vc["Persen"].apply(lambda x: f"{str(x).replace('.', ',')}%")

    custom_colors = alt.Scale(
        domain=["Negatif", "Positif", "Netral"],
        range=["#ff9bb8", "#8ab6ff", "#d9d9d9"]
    )

    pie = (
        alt.Chart(vc)
        .transform_joinaggregate(Total="sum(Jumlah)")
        .transform_calculate(Percent="datum.Jumlah / datum.Total")
        .mark_arc()
        .encode(
            theta="Jumlah:Q",
            color=alt.Color("Kategori:N", scale=custom_colors),
            tooltip=["Kategori:N", alt.Tooltip("Jumlah:Q", format=",.0f"), "PersenStr:N"],
        )
        .properties(width=320, height=320)
    )

    # ================= LABEL NORMAL (77,3% dan 16,5%) =================
    labels_main = (
        alt.Chart(vc[vc["Kategori"] != "Netral"])
        .transform_joinaggregate(Total="sum(Jumlah)")
        .transform_calculate(Percent="datum.Jumlah / datum.Total")
        .mark_text(
            fontSize=14,
            fontWeight="bold",
            color="black",
            radius=90     # radius default
        )
        .encode(
            theta="Jumlah:Q",
            text="PersenStr:N"
        )
    )

    # ================= LABEL NETRAL (6,2%) NAIK SEDIKIT =================
    labels_netral = (
        alt.Chart(vc[vc["Kategori"] == "Netral"])
        .transform_joinaggregate(Total="sum(Jumlah)")
        .transform_calculate(Percent="datum.Jumlah / datum.Total")
        .mark_text(
            fontSize=14,
            fontWeight="bold",
            color="black",
            radius=105     # dinaikkan sedikit ✔
        )
        .encode(
            theta="Jumlah:Q",
            text="PersenStr:N"
        )
    )

    # TENGAHKAN PIE
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        st.altair_chart(pie + labels_main + labels_netral, use_container_width=False)

    # # ================= WORDCLOUD =================
    # st.markdown(
    #     "<h4 style='font-size:20px; margin-top:25px;'>☁️ WordCloud per Kategori Sentimen</h4>",
    #     unsafe_allow_html=True
    # )

    # from wordcloud import WordCloud
    # import matplotlib.pyplot as plt

    # def _wc_fig(texts, title):
    #     wc = WordCloud(
    #         width=1200, height=650, background_color="white"
    #     ).generate(" ".join(texts))
    #     fig, ax = plt.subplots(figsize=(6.2, 3.6))
    #     ax.imshow(wc)
    #     ax.axis("off")
    #     ax.set_title(title)
    #     fig.tight_layout()
    #     return fig

    text_col = (
        "tokens_join"
        if "tokens_join" in df_lab.columns
        else ("content_norm" if "content_norm" in df_lab.columns else "content")
    )

    texts_pos = df_lab.loc[df_lab["label"] == 1, text_col].dropna().astype(str).tolist()
    texts_neg = df_lab.loc[df_lab["label"] == -1, text_col].dropna().astype(str).tolist()

    # c1, c2 = st.columns(2)
    # with c1:
    #     st.pyplot(_wc_fig(texts_pos, "WordCloud Positif"))
    # with c2:
    #     st.pyplot(_wc_fig(texts_neg, "WordCloud Negatif"))

    # ================= TOP 15 FREKUENSI =================
    st.markdown(
        "<h4 style='font-size:20px; margin-top:35px;'>📊 Top-15 Frekuensi Kata per Sentimen</h4>",
        unsafe_allow_html=True
    )

    import re
    from collections import Counter

    def top15_bar(texts, title):
        words = []
        for t in texts:
            words += re.findall(r"[a-zA-Z]{3,}", t.lower())

        top = Counter(words).most_common(15)
        if not top:
            st.warning(f"Tidak ada data untuk {title}")
            return

        wds, cts = zip(*top)
        y_pos = list(range(len(wds)))[::-1]
        vals = list(cts)[::-1]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(y_pos, vals)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(wds[::-1])
        ax.set_xlabel("Frekuensi")
        ax.set_title(title)

        def indo(n):
            return f"{n:,}".replace(",", ".")

        offset = max(vals) * 0.02
        for y, v in zip(y_pos, vals):
            ax.text(v + offset, y, indo(v), va="center", fontsize=10)

        st.pyplot(fig)

    colA, colB = st.columns(2)
    with colA:
        top15_bar(texts_pos, "Top-15 Kata Positif")
    with colB:
        top15_bar(texts_neg, "Top-15 Kata Negatif")

    st.divider()

        # # ================= TAHAP 4 =================
        # st.subheader("Tahap 4 — TF-IDF")
        # info_text(
        #     "Tahap ini menghitung bobot TF-IDF untuk menentukan kepentingan kata, "
        #     "di mana TF menunjukkan frekuensi kemunculan kata dalam dokumen, "
        #     "sedangkan IDF menunjukkan seberapa jarang atau pentingnya kata "
        #     "dalam seluruh dokumen."
        # )

        # tfidf_path = "data/tfidf_features_app_source_getcontact.csv"

        # if btn("Jalankan TF-IDF", key="t4_tfidf"):
        #     fn = getattr(mod, "stage7_tfidf", None)
        #     if not fn:
        #         st.error("Fungsi stage7_tfidf tidak ditemukan.")
        #     else:
        #         with st.spinner("Menghitung TF-IDF..."):
        #             out_feat = fn()

        #         st.session_state["res_tfidf_feat"] = out_feat
        #         st.success(f"✅ Selesai: {os.path.basename(out_feat)}")
        #         save_state()
        #         st.rerun()

    # ================= TAHAP 4 =================
    st.subheader("Tahap 4 — TF-IDF")
    info_text(
        "Tahap ini menghitung bobot TF-IDF untuk menentukan kepentingan kata, "
        "di mana TF menunjukkan frekuensi kemunculan kata dalam dokumen, "
        "sedangkan IDF menunjukkan seberapa jarang atau pentingnya kata "
        "dalam seluruh dokumen."
    )

    tfidf_path = "data/tfidf_features_app_source_getcontact.csv"

    # ----------------- TAMPILKAN HASIL -----------------
    if os.path.exists(tfidf_path):

        st.caption(f"TF-IDF Features: {os.path.basename(tfidf_path)}")

        try:
            df_feat = pd.read_csv(
                tfidf_path,
                usecols=["term", "idf", "tfidf"],
                dtype={"term": str, "idf": float, "tfidf": float}
            )

            label_file = latest(os.path.join(DATA_DIR, "label_only_*.csv"))
            docs_n = len(load_csv(label_file)) if label_file else 0
            uniq_n = len(df_feat)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Jumlah dokumen (TF)", f"{docs_n:,}")
            with col2:
                st.metric("Jumlah fitur unik (IDF)", f"{uniq_n:,}")

            # ----------- PENJELASAN IDF --------------
            st.markdown(
                "<h4 style='font-size:20px; margin-top:25px;'>🔹 Apa itu IDF?</h4>",
                unsafe_allow_html=True
            )

            st.markdown("""
            **Inverse Document Frequency (IDF)** adalah ukuran *seberapa jarang*
            sebuah kata muncul di seluruh dokumen.

            • IDF tinggi → kata **jarang muncul**  
            • IDF rendah → kata **sering muncul**
            """)

            # ----------- TABEL IDF -------------
            if "idf" in df_feat.columns and "term" in df_feat.columns:

                top_hi = df_feat.sort_values("idf", ascending=False).head(10)
                top_lo = df_feat.sort_values("idf", ascending=True).head(10)

                st.markdown(
                    "<h4 style='font-size:20px; margin-top:25px;'>📈 10 Kata dengan IDF Tertinggi</h4>",
                    unsafe_allow_html=True
                )
                st.table(top_hi[["term", "idf"]])

                st.markdown(
                    "<h4 style='font-size:20px; margin-top:25px;'>📉 10 Kata dengan IDF Terendah</h4>",
                    unsafe_allow_html=True
                )
                st.table(top_lo[["term", "idf"]])

        except Exception:
            st.error("Gagal memuat TF-IDF.")
            st.stop()

        # ----------- PENJELASAN TF-IDF -------------
        st.markdown(
            "<h4 style='font-size:20px; margin-top:25px;'>🔹 Apa itu TF-IDF?</h4>",
            unsafe_allow_html=True
        )

        st.markdown("""
        **TF-IDF (Term Frequency – Inverse Document Frequency)** digunakan untuk
        mengukur *seberapa penting* sebuah kata dalam suatu dokumen.
        
        Nilai TF-IDF tinggi berarti:

        • TF tinggi → kata **sering muncul** pada dokumen itu
        
        • IDF tinggi → kata **jarang muncul** di dokumen lain
        """)

        st.markdown(
            "<h4 style='font-size:20px; margin-top:25px;'>📊 Top 10 Kata dengan Nilai TF-IDF Tertinggi</h4>",
            unsafe_allow_html=True
        )
        st.markdown("Menampilkan daftar kata dengan skor TF-IDF terbesar.")

        if "tfidf" in df_feat.columns:
            top_tfidf = df_feat.sort_values("tfidf", ascending=False).head(10)
            st.table(top_tfidf[["term", "tfidf"]])

        st.markdown(
            make_download_button(
                tfidf_path,
                "Unduh TF-IDF",
                os.path.basename(tfidf_path)
            ),
            unsafe_allow_html=True
        )

    else:
        st.error("File TF-IDF tidak ditemukan! Pastikan file berada di folder /data/")

    st.divider()

    # ================= TAHAP 5 =================
    st.subheader("Tahap 5 — Pembagian Data Latih dan Data Uji")
    info_text(
        "Tahap ini menggunakan matriks TF-IDF untuk membagi data ke lima rasio: 90/10, 80/20, 70/30, 60/40, dan 50/50. "
        "Data latih digunakan untuk melatih model mengenali pola, sedangkan data uji digunakan untuk mengevaluasi prediksi model pada data baru."
    )

    # # ======= BUTTON SPLIT =======
    # if btn("⚡ Jalankan 5 Rasio Split", key="t5_multi_split"):

    #     if not st.session_state.get("res_tfidf_feat"):
    #         st.error("TF-IDF belum dijalankan!")
    #         st.stop()

    #     # tempat menampilkan status berjalan
    #     live = st.empty()

    #     ratios = {
    #         "90_10": 0.90,
    #         "80_20": 0.80,
    #         "70_30": 0.70,
    #         "60_40": 0.60,
    #         "50_50": 0.50,
    #     }

    #     results = {}

    #     for key, frac in ratios.items():

    #         # update status UI
    #         live.info(f"🔄 Memproses rasio {key.replace('_','/')} ...")

    #         # jalankan split untuk rasio ini
    #         fn = getattr(mod, "stage8_split", None)

    #         if not fn:
    #             st.error("Fungsi stage8_split tidak ditemukan di pipeline.py")
    #             st.stop()

    #         out_train, out_test, train_n, test_n = fn(frac)

    #         # simpan hasil
    #         results[key] = {
    #             "train": out_train,
    #             "test": out_test,
    #             "train_n": train_n,
    #             "test_n": test_n,
    #         }

    #     live.success("✅ Semua rasio selesai dibuat!")

    #     st.session_state["res_split_multi"] = results
    #     save_state()
    #     st.rerun()


    # ========== HASIL SPLIT ==========
    if st.session_state.get("res_split_multi"):
        data = st.session_state["res_split_multi"]

        rows = []
        for key, d in data.items():
            rows.append([key.replace("_", "/"), d["train_n"], d["test_n"]])

        df_info = pd.DataFrame(
            rows,
            columns=["Rasio", "Jumlah Data Latih", "Jumlah Data Uji"]
        )

        st.markdown(
            "<h4 style='font-size:20px; margin-top:25px;'>📋 Tabel Rasio</h4>",
            unsafe_allow_html=True
        )
        st.dataframe(df_info, use_container_width=True)

        st.markdown(
            "<h4 style='font-size:20px; margin-top:25px;'>📊 Visualisasi Pembagian Data Latih dan Data Uji</h4>",
            unsafe_allow_html=True
        )

        chart_data = df_info.rename(columns={
            "Jumlah Data Latih": "Train",
            "Jumlah Data Uji": "Test"
        })

        melted = chart_data.melt(
            id_vars="Rasio",
            value_vars=["Train", "Test"],
            var_name="Jenis",
            value_name="Jumlah"
        )

        melted["JumlahLabel"] = melted["Jumlah"].apply(lambda x: f"{x:,}".replace(",", "."))

        bars = (
            alt.Chart(melted)
            .mark_bar()
            .encode(
                x=alt.X("Rasio:N", title="Rasio", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Jumlah:Q", title="Jumlah Dokumen",
                        axis=alt.Axis(format=",.0f")),  
                color=alt.Color(
                    "Jenis:N",
                    scale=alt.Scale(
                        domain=["Train", "Test"],     
                        range=["#8ab6ff", "#ff9bb8"] 
                    ),
                    title="Kategori"
                ),
                xOffset=alt.XOffset("Jenis:N", sort=["Train", "Test"]),
                tooltip=["Rasio", "Jenis", "JumlahLabel"]
            )
        )

        labels = (
            alt.Chart(melted)
            .mark_text(
                dy=-5,
                fontSize=12,
                color="black"
            )
            .encode(
                x=alt.X("Rasio:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Jumlah:Q"),
                xOffset=alt.XOffset("Jenis:N", sort=["Train", "Test"]),
                text=alt.Text("JumlahLabel:N")  
            )
        )

        final_chart = (bars + labels).properties(
            width="container",
            height=350
        )

        st.altair_chart(final_chart, use_container_width=True)

        # st.markdown(
        #     "<h4 style='font-size:20px; margin-top:10px;'>📥 Unduhan File</h4>",
        #     unsafe_allow_html=True
        # )

        # for key, d in data.items():
        #     st.markdown(
        #         f"<h5 style='font-size:18px; margin-top:5px;'>🔸 Rasio {key.replace('_','/')}</h5>",
        #         unsafe_allow_html=True
        #     )

        #     col1, col2 = st.columns(2)

        #     with col1:
        #         st.markdown(
        #             make_download_button(
        #                 d["train"],
        #                 f"Train CSV — Rasio {key.replace('_','/')}",
        #                 os.path.basename(d["train"])
        #             ),
        #             unsafe_allow_html=True
        #         )

        #     with col2:
        #         st.markdown(
        #             make_download_button(
        #                 d["test"],
        #                 f"Test CSV — Rasio {key.replace('_','/')}",
        #                 os.path.basename(d["test"])
        #             ),
        #             unsafe_allow_html=True
        #         )

    # ================= TAHAP 6 =================
    st.subheader("Tahap 6 — Klasifikasi SVM")
    info_text(
        "Kernel Linear dan Kernel RBF untuk menjalankan klasifikasi pada semua rasio "
        "(90/10, 80/20, 70/30, 60/40, 50/50). "
        "Data latih diproses dengan hyperparameter tuning (GridSearchCV) "
        "untuk memperoleh parameter terbaik, sedangkan data uji digunakan "
        "untuk evaluasi melalui confusion matrix dan classification report."
    )

    # ================= FUNGSI UNTUK TIAP RASIO =================
    def run_svm_linear_for_ratio(ratio_key):
        info = st.session_state['res_split_multi'][ratio_key]
        mod.TRAIN_FILE = info['train']
        mod.TEST_FILE = info['test']

        (
            rep,
            fig,
            cm_df,
            best_params,
            best_score,
            cv_results,
            cv_used
        ) = mod.stage9_svm_linear()

        return {
            "ratio": ratio_key.replace("_", "/"),
            "rep": rep,
            "fig": fig,
            "cm_df": cm_df,
            "best_params": best_params,
            "best_score": best_score,
            "cv_results": cv_results,
            "cv_used": cv_used
        }

    def run_svm_rbf_for_ratio(ratio_key):
        info = st.session_state['res_split_multi'][ratio_key]
        mod.TRAIN_FILE = info['train']
        mod.TEST_FILE = info['test']

        progress = st.empty()

        def _cb(msg):
            progress.markdown(
                f"""
                <div style='padding:8px; background:#fff7ed; border:1px solid #fdba74; border-radius:8px; margin:6px 0;'>
                    {msg}
                </div>
                """,
                unsafe_allow_html=True
            )

        (
            rep,
            fig,
            cm_df,
            best_params,
            best_score,
            cv_results,
            cv_used
        ) = mod.stage10_svm_rbf(progress_callback=_cb)

        progress.empty()

        return {
            "ratio": ratio_key.replace("_", "/"),
            "rep": rep,
            "fig": fig,
            "cm_df": cm_df,
            "best_params": best_params,
            "best_score": best_score,
            "cv_results": cv_results,
            "cv_used": cv_used
        }

    # # ================= BUTTON — SVM LINEAR =================
    # if btn("1️⃣ Jalankan SVM Linear", key="t6_svm_linear_multi"):
    #     if not st.session_state.get("res_split_multi"):
    #         st.error("Split belum dijalankan!")
    #         st.stop()

    #     results = []
    #     status_box = st.empty()

    #     with st.spinner("Menjalankan SVM Linear untuk semua rasio..."):
    #         for key in ["90_10", "80_20", "70_30", "60_40", "50_50"]:

    #             ratio_text = key.replace("_", "/")

    #             status_box.markdown(
    #                 f"""
    #                 <div style='padding:10px; background:#eef2ff; border-radius:8px;
    #                 border:1px solid #c7d2fe; margin:10px 0;'>
    #                     🔄 <b>Memproses data latih rasio {ratio_text}...</b>
    #                 </div>
    #                 """,
    #                 unsafe_allow_html=True
    #             )

    #             results.append(run_svm_linear_for_ratio(key))

    #             status_box.markdown(
    #                 f"""
    #                 <div style='padding:10px; background:#e0f2fe; border-radius:8px;
    #                 border:1px solid #7dd3fc; margin:10px 0;'>
    #                     🔄 <b>Memproses data uji rasio {ratio_text}...</b>
    #                 </div>
    #                 """,
    #                 unsafe_allow_html=True
    #             )

    #     status_box.empty()
    #     st.session_state["svm_linear_multi"] = results
    #     st.success("✅ SVM Linear selesai untuk semua rasio")
    #     save_state()
    #     st.rerun()

    # # ===================== BUTTON SVM RBF JIKA LINEAR BELUM JALAN =====================
    # if not st.session_state.get("svm_linear_multi"):
    #     if btn("2️⃣ Jalankan SVM RBF", key="t6_svm_rbf_top"):

    #         results = []
    #         status_box = st.empty()

    #         with st.spinner("Melatih SVM RBF untuk semua rasio..."):
    #             for key in ["90_10", "80_20", "70_30", "60_40", "50_50"]:

    #                 ratio_text = key.replace("_", "/")

    #                 status_box.markdown(
    #                     f"""
    #                     <div style='padding:10px; background:#eef2ff; border-radius:8px;
    #                     border:1px solid #c7d2fe; margin:10px 0;'>
    #                         🔄 <b>Memproses data latih rasio {ratio_text}...</b>
    #                     </div>
    #                     """,
    #                     unsafe_allow_html=True
    #                 )

    #                 results.append(run_svm_rbf_for_ratio(key))

    #                 status_box.markdown(
    #                     f"""
    #                     <div style='padding:10px; background:#e0f2fe; border-radius:8px;
    #                     border:1px solid #7dd3fc; margin:10px 0;'>
    #                         🔄 <b>Memproses data uji rasio {ratio_text}...</b>
    #                     </div>
    #                     """,
    #                     unsafe_allow_html=True
    #                 )

    #         status_box.empty()
    #         st.session_state["svm_rbf_multi"] = results
    #         save_state()
    #         st.success("✅ SVM RBF selesai untuk semua rasio")
    #         st.rerun()

    # ===================== TAMPILKAN HASIL SVM LINEAR =====================
    if st.session_state.get("svm_linear_multi"):
        st.markdown(
            "<h3 style='margin-top:20px; font-size:20px; font-weight:600;'>📊 Hasil Lengkap — SVM Linear</h3>",
            unsafe_allow_html=True
        )
        for r in st.session_state["svm_linear_multi"]:
            ratio = r["ratio"]
            st.markdown(
                f"<h4 style='margin-top:15px; font-size:17px;'>🔹 Rasio {ratio}</h4>",
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <div style='padding:10px; background:#fafafa; border-radius:8px; border:1px solid #ddd; margin-bottom:8px;'>
                    <h5 style='margin:0; padding-bottom:4px; font-size:15px;'>📄 Confusion Matrix</h5>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.dataframe(r["cm_df"], use_container_width=True)

            st.markdown(
                """
                <div style='padding:10px; background:#fafafa; border-radius:8px; border:1px solid #ddd; margin-top:10px; margin-bottom:8px;'>
                    <h5 style='margin:0; padding-bottom:4px; font-size:15px;'>📄 Classification Report</h5>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.code(r["rep"], language="text")
            st.markdown("---")

        # st.markdown(
        #     "<p style='font-size:15px; margin-top:15px;'>Setelah model Linear selesai, jalankan model RBF:</p>",
        #     unsafe_allow_html=True
        # )

        # if btn("2️⃣ Jalankan SVM RBF", key="t6_svm_rbf_bottom"):
        #     results = []
        #     with st.spinner("Melatih SVM RBF untuk semua rasio..."):
        #         for key in ["90_10", "80_20", "70_30", "60_40", "50_50"]:
        #             results.append(run_svm_rbf_for_ratio(key))

        #     st.session_state["svm_rbf_multi"] = results
        #     save_state()
        #     st.success("✅ SVM RBF selesai untuk semua rasio")
        #     st.rerun()

    # ===================== TAMPILKAN HASIL SVM RBF =====================
    if st.session_state.get("svm_rbf_multi"):

        st.markdown(
            "<h3 style='margin-top:20px; font-size:20px; font-weight:600;'>📊 Hasil Lengkap — SVM RBF</h3>",
            unsafe_allow_html=True
        )

        for r in st.session_state["svm_rbf_multi"]:
            ratio = r["ratio"]

            st.markdown(
                f"<h4 style='margin-top:15px; font-size:17px;'>🔹 Rasio {ratio}</h4>",
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div style='padding:10px; background:#fafafa; border-radius:8px;
                border:1px solid #ddd; margin-bottom:8px;'>
                    <h5 style='margin:0; padding-bottom:4px; font-size:15px;'>🔷 Confusion Matrix</h5>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.dataframe(r["cm_df"], use_container_width=True)

            st.markdown(
                """
                <div style='padding:10px; background:#fafafa; border-radius:8px;
                border:1px solid #ddd; margin-top:10px; margin-bottom:8px;'>
                    <h5 style='margin:0; padding-bottom:4px; font-size:15px;'>📄 Classification Report</h5>
                </div>
                """,
                unsafe_allow_html=True
            )

            bestC = r["best_params"].get("C", "?")
            bestG = r["best_params"].get("gamma", "?")

            st.code(
                f"Model: SVC(kernel='rbf')\n"
                f"K-Fold CV: {r['cv_used']}\n"
                f"Best C: {bestC}\n"
                f"Best gamma: {bestG}\n\n"
                + r["rep"],
                language="text"
            )

        st.markdown("---")
