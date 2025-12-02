import streamlit as st
import pandas as pd
import re
from helpers import load_csv, DATA_DIR, latest

def render():

    st.markdown("<h1 style='text-align:center;'>Analisis Permasalahan Berdasarkan Ulasan Pengguna</h1>", unsafe_allow_html=True)

    st.write("""
    Halaman ini dibuat untuk menganalisis apakah aplikasi **Getcontact** benar-benar membantu 
    mengurangi spam call, meningkatkan keamanan pengguna, serta apakah fitur-fitur seperti 
    **tagar** bekerja secara akurat.

    Permasalahan yang dianalisis berasal dari **Latar Belakang Penelitian**, yaitu:
    """)

    st.markdown("""
    ### 🔹 *1. Ancaman spam call & penipuan (Hiya, 2023–2025)*
    Indonesia menjadi negara dengan spam call tertinggi di Asia Pasifik.  
    Spam rate: **61% (2023) → 82% (2024) → 89% (2025)**.  
    → **Permasalahan:** apakah Getcontact benar-benar membantu memblokir spam dan penipuan?

    ### 🔹 *2. Banyaknya kebocoran data pribadi (Medcom, RMOL – 2024)*
    Berulang kali terjadi kebocoran data besar dari lembaga pemerintah & swasta.  
    → **Permasalahan:** apakah pengguna merasa aman? apakah ada komentar soal data bocor, privasi, atau ketidakpercayaan?

    ### 🔹 *3. Getcontact populer tetapi keamanannya diperdebatkan*
    Aplikasi mengunggah buku kontak, tagar tidak akurat, risiko penyalahgunaan tag name.  
    → **Permasalahan:** apakah pengguna mengalami ketidakakuratan tagar atau khawatir datanya disalahgunakan?
    """)

    st.divider()

    # ===============================
    #  Load data label Only (hasil InSet)
    # ===============================
    path = latest(f"{DATA_DIR}/label_only_app_source_getcontact.csv")

    if not path:
        st.error("File label tidak ditemukan.")
        return
    
    df = load_csv(path)
    
    text_col = "content_norm" if "content_norm" in df.columns else "content"

    st.success(f"Data berhasil dimuat: {len(df):,} ulasan")

    # ===============================
    #  PILIH PERMASALAHAN
    # ===============================
    st.markdown("### 🎯 Pilih Permasalahan yang Ingin Dianalisis")
    pilihan = st.selectbox(
        "Kategori Permasalahan:",
        [
            "Spam Call & Penipuan",
            "Keamanan Data & Privasi",
            "Tagar Tidak Akurat & Penyalahgunaan Data"
        ]
    )

    # ===============================
    #  DEFINISI KATA KUNCI PALING RELEVAN
    # ===============================

    patterns = {
        "Spam Call & Penipuan": 
            r"spam|scam|penipu|penipuan|tipu|nipu|telepon|panggilan|blokir|call",

        "Keamanan Data & Privasi": 
            r"aman|keamanan|privasi|privacy|data|bocor|kebocoran|virus|trojan|malware|bahaya|berbahaya|curi|ambil data",

        "Tagar Tidak Akurat & Penyalahgunaan Data":
            r"tagar|tag name|tidak akurat|tidak sesuai|tidak muncul|salah|fitnah|penyalahgunaan"
    }

    regex = patterns[pilihan]

    st.info(f"🔍 Kata kunci pencarian: `{regex}`")

    # ===============================
    #  FILTER BERDASARKAN KATA KUNCI
    # ===============================

    df_match = df[df[text_col].str.contains(regex, case=False, regex=True)]

    st.metric("Jumlah Ulasan Relevan", f"{len(df_match):,}")

    if df_match.empty:
        st.warning("Tidak ada ulasan yang relevan ditemukan.")
        return

    # ===============================
    #  TAMPILKAN TOP KATA KUNCI (berdasarkan hasil pencarian)
    # ===============================

    st.markdown("### 🔎 Kata yang Paling Sering Muncul di Ulasan Relevan")

    # hanya ambil kata yang relevan (yang memuat kata kunci pencarian)
    relevant_words = []

    for text in df_match[text_col]:
        found = re.findall(regex, text, flags=re.IGNORECASE)
        relevant_words.extend([f.lower() for f in found])

    freq = pd.Series(relevant_words).value_counts().head(15)

    st.dataframe(freq.to_frame("Frekuensi"), use_container_width=True)

    st.markdown("### 💬 Ulasan Pengguna")

    for _, row in df_match.head(100).iterrows():
        st.write(f"- {row[text_col]}")

    st.caption("Menampilkan 100 ulasan pertama yang relevan.")

    st.divider()
