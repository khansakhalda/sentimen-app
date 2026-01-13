# 📊 Aplikasi Analisis Sentimen Ulasan Getcontact

Aplikasi ini merupakan **aplikasi berbasis web menggunakan Streamlit** yang dikembangkan untuk
melakukan **analisis sentimen ulasan pengguna aplikasi Getcontact** yang diperoleh dari
Google Play Store.

Penelitian ini bertujuan untuk memahami **persepsi pengguna terhadap Getcontact**, khususnya
terkait efektivitas pemblokiran spam, keamanan data dan privasi, serta akurasi fitur tagar
(tag name) yang disediakan oleh aplikasi tersebut.

Aplikasi ini dikembangkan sebagai bagian dari **Tugas Akhir / Skripsi**.

---

## 🎯 Tujuan Aplikasi
Aplikasi ini dibuat untuk:
- Mengklasifikasikan ulasan pengguna ke dalam **sentimen positif dan negatif**
- Menganalisis **tren sentimen pengguna dari waktu ke waktu (per kuartal)**
- Mengidentifikasi **permasalahan utama pengguna** seperti:
  - Spam call & penipuan
  - Keamanan data & privasi
  - Ketidakakuratan tagar (tag name)
- Menampilkan hasil analisis secara **interaktif dan visual**

---

## 🧠 Metode yang Digunakan
Aplikasi ini menggunakan kombinasi metode **Lexicon-Based** dan **Machine Learning**, yaitu:

- **InSet Lexicon**  
  Digunakan untuk pelabelan otomatis sentimen ulasan  
  - Skor > 0 → Positif  
  - Skor < 0 → Negatif  
  - Skor = 0 → Netral (tidak dianalisis lebih lanjut)

- **TF-IDF (Term Frequency – Inverse Document Frequency)**  
  Digunakan untuk ekstraksi fitur teks

- **Support Vector Machine (SVM)**  
  - Kernel Linear  
  - Kernel RBF  
  Evaluasi dilakukan dengan berbagai rasio data latih dan data uji

- **Analisis Tren Sentimen**  
  Berdasarkan pembagian waktu per kuartal

---

## 🧩 Fitur Utama Aplikasi

### 1️⃣ Deskripsi Analisis
Menjelaskan:
- Latar belakang penelitian
- Konteks penggunaan aplikasi Getcontact
- Permasalahan pengguna
- Ruang lingkup dan tahapan analisis sentimen

---

### 2️⃣ Proses Analisis
Menampilkan tahapan analisis secara bertahap, meliputi:
1. Pengumpulan data ulasan Google Play Store  
2. Text preprocessing (cleaning, case folding, normalisasi, tokenisasi)  
3. Pelabelan sentimen menggunakan InSet Lexicon  
4. Ekstraksi fitur TF-IDF  
5. Pembagian data latih & data uji  
6. Klasifikasi menggunakan SVM Linear dan RBF  
7. Evaluasi performa model (confusion matrix & classification report)

---

### 3️⃣ Pencarian Sentimen
Fitur untuk:
- Mencari ulasan berdasarkan kata atau frasa tertentu
- Menyaring hasil berdasarkan sentimen positif atau negatif
- Menampilkan ulasan secara terstruktur dan interaktif

---

### 4️⃣ Analisis Tren Sentimen
Menampilkan:
- Perubahan sentimen pengguna dari waktu ke waktu
- Distribusi sentimen positif dan negatif per kuartal
- Ringkasan otomatis berbasis AI (Gemini) untuk interpretasi tren sentimen

---

### 5️⃣ Analisis Permasalahan
Menganalisis ulasan berdasarkan kategori permasalahan:
- Spam Call & Penipuan
- Keamanan Data & Privasi
- Tagar Tidak Akurat & Penyalahgunaan Data

Analisis dilakukan menggunakan pendekatan **filtering berbasis kata kunci (regex)**.

---

## 🚀 Cara Menjalankan Aplikasi

### ▶️ Menjalankan Secara Lokal

1. Clone repositori ini:
```bash
git clone https://github.com/khansakhalda/sentimen-app.git
cd sentimen-app
Install dependensi:

1. Clone repositori ini:
```bash
git clone https://github.com/khansakhalda/sentimen-app.git
cd sentimen-app
Install dependensi:
