import os
import streamlit as st
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import gdown
import pickle
from transformers import AutoModelForSequenceClassification
import io
import torch

# Paksa semua tensor yang di-load selalu ditaruh di CPU,
# apa pun device asal saat model disimpan
_original_load_from_bytes = torch.storage._load_from_bytes
torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")

# ============ KONFIGURASI HALAMAN ============
st.set_page_config(
    page_title="Klasifikasi Teks PPKD",
    page_icon="📋",
    layout="centered",
)

GDRIVE_FILE_ID = "1VJ5PSib6jv3S9YVd3WHpaFPSGXmFUk-w"   # <-- ganti dengan ID dari Langkah 2
MODEL_PATH = "indobert_ppkd_model.pkl"

# Warna tetap per kelas, dipakai di bar chart
LABEL_COLORS = {
    "apresiasi": "#2E8B57",
    "netral": "#5B7C99",
    "saran": "#E08E00",
    "keluhan": "#C0392B",
}

# ============ DOWNLOAD & LOAD MODEL (cache, hanya jalan sekali) ============
@st.cache_resource(show_spinner="Menyiapkan model, mohon tunggu sebentar...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = AutoModelForSequenceClassification.from_pretrained(
        artifact["model_name"], num_labels=len(artifact["label2id"])
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    return model, artifact["tokenizer"], artifact["id2label"]

model, tokenizer, id2label = load_model()

# ============ FUNGSI PREDIKSI ============
def predict(text):
    inputs = tokenizer(
        text, padding=True, truncation=True, max_length=64, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
    return {id2label[i]: float(probs[i]) for i in range(len(probs))}

# ============ CSS TAMBAHAN (tampilan modern) ============
st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #1E3A5F;
        color: white;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover { background-color: #2E5484; color: white; }
    .result-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        background-color: #F2F5F9;
        border-left: 6px solid #1E3A5F;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============ HEADER ============
st.markdown("<h1 style='color:#1E3A5F;'>📋 Klasifikasi Teks Sosialisasi PPKD</h1>", unsafe_allow_html=True)
st.markdown("Model IndoBERT untuk mengklasifikasikan masukan menjadi **apresiasi / netral / saran / keluhan**.")
st.divider()

# ============ INPUT ============
text_input = st.text_area(
    "Masukkan teks",
    placeholder="Contoh: Materi pelatihan sangat jelas dan mudah dipahami...",
    height=140,
)

predict_clicked = st.button("🔍 Prediksi")

# ============ OUTPUT ============
if predict_clicked:
    if not text_input.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Memproses..."):
            result = predict(text_input)

        top_label = max(result, key=result.get)
        top_score = result[top_label]

        st.markdown(
            f"""
            <div class="result-card">
                <span style="font-size:0.9rem;color:#555;">Hasil Prediksi</span><br>
                <span style="font-size:1.8rem;font-weight:700;color:{LABEL_COLORS.get(top_label,'#1E3A5F')};">
                    {top_label.upper()}
                </span>
                <span style="font-size:1.1rem;color:#333;"> — confidence {top_score*100:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Bar chart skor semua kelas
        sorted_items = sorted(result.items(), key=lambda x: x[1])
        labels_sorted = [k for k, v in sorted_items]
        scores_sorted = [v for k, v in sorted_items]
        colors_sorted = [LABEL_COLORS.get(k, "#1E3A5F") for k in labels_sorted]

        fig = go.Figure(go.Bar(
            x=scores_sorted,
            y=labels_sorted,
            orientation="h",
            marker_color=colors_sorted,
            text=[f"{s*100:.1f}%" for s in scores_sorted],
            textposition="outside",
        ))
        fig.update_layout(
            title="Skor Confidence per Kelas",
            xaxis=dict(range=[0, 1], tickformat=".0%"),
            height=320,
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "<hr><p style='text-align:center;color:#888;font-size:0.85rem;'>PPKD Text Classifier — IndoBERT</p>",
    unsafe_allow_html=True,
)