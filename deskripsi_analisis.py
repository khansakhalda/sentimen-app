import os
import pandas as pd
import streamlit as st

def _read_date_window():
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("pipe", "pipeline.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "DEFAULT_FROM_DATE", None), getattr(mod, "DEFAULT_TO_DATE", None)
    except Exception:
        return None, None

def render():
    st.markdown("<h1 style='text-align:center;'>Deskripsi Analisis — Getcontact</h1>", unsafe_allow_html=True)
    st.caption("Halaman ini menjelaskan konteks riset, fungsi aplikasi Getcontact, permasalahan pengguna, serta tahapan analisis yang dilakukan dalam aplikasi ini.")

    st.markdown("""
### 🌐 Konteks Penelitian
Perkembangan teknologi digital telah membawa banyak perubahan signifikan dalam kehidupan masyarakat. Dengan semakin luasnya penggunaan internet dan smartphone, aktivitas komunikasi menjadi lebih mudah, cepat, dan terhubung. Namun, di balik kemudahan tersebut, muncul tantangan baru berupa meningkatnya kejahatan siber, salah satunya melalui **panggilan spam dan penipuan telepon**.

Indonesia secara konsisten menempati posisi teratas di kawasan Asia Pasifik dalam laporan *Hiya Global Call Threat Report (2023–2025)*, dengan tingkat spam call yang terus meningkat dari tahun ke tahun, bahkan mencapai **89% pada awal 2025**. Temuan ini mencerminkan ancaman serius terhadap keamanan komunikasi digital di Indonesia.

Selain itu, maraknya **kebocoran data pribadi** turut memperkuat kekhawatiran masyarakat terhadap perlindungan data. Untuk menjawab permasalahan ini, masyarakat mulai beralih menggunakan aplikasi identifikasi nomor dan pemblokiran panggilan seperti **Getcontact**, yang menjadi objek penelitian dalam aplikasi analisis ini.
""")


    st.markdown("""
### 📱 Tentang Getcontact
**Getcontact** adalah aplikasi yang berfungsi untuk:
- **Mengidentifikasi nomor tidak dikenal**, termasuk panggilan dari penipu atau telemarketer.  
- **Memblokir panggilan spam** secara otomatis melalui sistem komunitas pengguna.  
- **Menampilkan tag nama** yang diberikan pengguna lain untuk suatu nomor.  
- Menyediakan fitur tambahan seperti:
  - **Spam Protection** dan *Trust Score* (skor kepercayaan nomor),
  - **Who Called / Search**, untuk melacak identitas penelepon secara real-time,
  - **Messenger**, dan
  - Dukungan **multiplatform** untuk kemudahan akses di berbagai perangkat.

Aplikasi ini telah diunduh lebih dari **100 juta kali** di Google Play Store dan digunakan oleh lebih dari **850 juta pengguna di seluruh dunia**, menjadikannya salah satu aplikasi *caller identification* dan *spam blocker* terpopuler.
""")

    st.markdown("""
### 🔒 Permasalahan Kepercayaan Pengguna
Walaupun populer, **Getcontact tidak terlepas dari isu privasi data dan kepercayaan pengguna**.  
Beberapa pengguna merasa terbantu karena fitur perlindungan spam, namun sebagian lain mengkhawatirkan keamanan data pribadi.  
Menurut beberapa laporan media:
- Aplikasi ini **meminta akses ke buku kontak pengguna**, lalu **mengunggah data tersebut ke server** untuk ditampilkan kepada pengguna lain tanpa persetujuan eksplisit.
- Informasi “tag nama” yang muncul bisa **tidak akurat** atau **disalahgunakan oleh pihak ketiga**.

Masalah ini relevan dengan meningkatnya kasus kebocoran data pribadi di Indonesia, yang menimbulkan keresahan publik dan menurunkan tingkat kepercayaan digital masyarakat.
""")

    st.markdown("""
### 🎯 Alasan Pemilihan Objek Penelitian
Getcontact dipilih sebagai objek analisis karena:
1. Memiliki **basis pengguna besar dan aktif** di Indonesia.  
2. Mengandung **isu sosial dan teknologi** penting, antara lain keamanan data, kepercayaan digital, dan persepsi publik terhadap layanan privasi.  
3. Banyaknya **ulasan pengguna di Google Play Store** memungkinkan analisis sentimen berbasis data tekstual yang representatif terhadap opini masyarakat.
""")

    st.markdown("""
### 🎓 Tujuan Penelitian
- Mengidentifikasi persepsi pengguna terhadap **Getcontact** berdasarkan ulasan publik di Google Play Store.  
- Mengklasifikasikan opini pengguna menjadi **sentimen positif dan negatif** menggunakan pendekatan **InSet Lexicon** dan algoritma **Support Vector Machine (SVM)**.  
- Membandingkan performa dua jenis **fungsi kernel** (Linear dan RBF) untuk menentukan model klasifikasi terbaik.
""")

    st.markdown("""
### 📦 Ruang Lingkup
- **Objek:** Aplikasi Getcontact  
- **Sumber Data:** Ulasan pengguna di Google Play Store  
- **Bahasa:** Indonesia (id-ID)  
- **Periode Data:** 1 September 2024 – 31 Agustus 2025  
- **Metode:** Pelabelan InSet Lexicon dan Klasifikasi SVM (Kernel Linear & RBF)
""")

    st.markdown("""
### 🔗 Pipeline Analisis Sentimen
1. **Pengumpulan Data** = Data dikumpulkan dari ulasan pengguna aplikasi Getcontact di Google Play Store menggunakan pustaka `google-play-scraper`.  
2. **Prapemrosesan Teks** = Tahapan ini mencakup pembersihan karakter tidak relevan, case folding (mengubah huruf menjadi huruf kecil), normalisasi kata tidak baku, dan tokenisasi.  
3. **Pelabelan Otomatis** = Proses pelabelan dilakukan secara otomatis menggunakan **InSet Lexicon**, dengan ketentuan: skor > 0 → positif, skor < 0 → negatif.  
4. **Ekstraksi Fitur** = Data ulasan yang telah diproses diubah menjadi vektor numerik menggunakan metode **TF-IDF (Term Frequency–Inverse Document Frequency)**.  
5. **Pembagian Data** = Dataset dibagi ke dalam lima variasi rasio data latih dan data uji, yaitu: **90:10**, **80:20**, **70:30**, **60:40**, dan **50:50**.  
6. **Klasifikasi Sentimen** = Model dibangun menggunakan algoritma **Support Vector Machine (SVM)**, dengan dua jenis kernel yang diuji: **Linear** dan **RBF**.  
7. **Evaluasi Model** = Performa model dievaluasi menggunakan metrik **accuracy**, **precision**, **recall**, dan **F1-score**, serta divisualisasikan dalam bentuk **confusion matrix** dan classification report.
""")

    st.markdown("""
### ▶️ Panduan Penggunaan Aplikasi
1. Gunakan menu **Proses Analisis** untuk melihat tahap 1–6 secara berurutan.  
3. Gunakan menu **Pencarian Sentimen** untuk menelusuri ulasan berdasarkan kata kunci.  
4. Gunakan menu **Analisis Tren Sentimen** untuk melihat perubahan persepsi pengguna per kuartal.
""")

    with st.expander("⚠️ Batasan Analisis"):
        st.markdown("""
- **Klasifikasi Biner:** Model hanya mengenali dua kategori utama sentimen, yaitu “positif” dan “negatif”. 
- **Label InSet** menggunakan ambang: `score > 0 → positif`, `< 0 → negatif` (skor 0 dianggap netral, tapi tidak sertakan dalam analisis).  
- **Ketergantungan leksikon:** Akurasi bergantung pada keluasan kata dalam InSet Lexicon.  
- **Keterbatasan Bahasa:** Analisis hanya mencakup ulasan berbahasa Indonesia.   
""")
