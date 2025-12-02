import streamlit as st

# judul halaman
st.set_page_config(
    page_title="Analisis Sentimen Ulasan",
    layout="wide"
)

# Import modul pendukung dan halaman
from helpers import init_state
import deskripsi_analisis as page_desc
import proses_analisis as page_proses
import pencarian_sentimen as page_cari
import analisis_tren as page_tren
import analisis_permasalahan as page_permasalahan

# Muat state lama (jika ada)
# Bagian ini wajib dipanggil paling awal untuk mengembalikan
# state sebelumnya sehingga data tidak hilang saat berpindah halaman
page_proses.load_state()

# Inisialisasi state default
# Dilakukan hanya sekali di awal agar tidak menimpa state yang sudah ada
if "initialized" not in st.session_state:
    init_state()
    st.session_state["initialized"] = True

# sidebar nav
with st.sidebar:
    st.image("getcontact.png")
    st.radio(
        " ",
        ["Deskripsi Analisis", "Proses Analisis", "Pencarian Sentimen", "Analisis Tren Sentimen", "Analisis Permasalahan"],
        key="page" # simpan pilihan ke session_state
    )
    st.divider()

# router
page = st.session_state.get("page", "Deskripsi Analisis")

if page == "Deskripsi Analisis":
    page_desc.render()
elif page == "Pencarian Sentimen":
    page_cari.render()
elif page == "Analisis Tren Sentimen":
    page_tren.render()
elif page == "Analisis Permasalahan":
    page_permasalahan.render()
else:
    page_proses.render()
