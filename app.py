import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
import ssl

# ==========================================
# FIX SSL
# ==========================================
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# ==========================================
# CONFIGURAÇÃO
# ==========================================
st.set_page_config(page_title="ALPR Alta Precisão", layout="centered")

@st.cache_resource
def load_models():
    yolo = YOLO('yolov8n.pt')
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return yolo, reader

def preprocess(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Reforço de contraste para placas em condições de rua
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    return clahe.apply(gray)

def extract_best_plate(image_np, yolo_model, ocr_reader):
    # Processa imagem inteira (mais robusto para ângulos e distâncias)
    img_prep = preprocess(image_np)
    results = ocr_reader.readtext(img_prep)
    
    # Extrai todos os textos e suas confianças
    candidates = []
    for bbox, text, conf in results:
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(clean) >= 3:
            candidates.append({'text': clean, 'conf': conf})

    # Tenta montar placas validando padrões 
    # (Padrão Antigo: 3 letras + 4 números | Mercosul: 3 letras + 1 num + 1 let + 2 num)
    pattern = re.compile(r'([A-Z]{3})([0-9][A-Z0-9][0-9]{2})')
    
    best_match = None
    max_conf = 0
    
    # Busca padrões em todos os textos lidos
    for cand in candidates:
        match = pattern.search(cand['text'])
        if match:
            plate = f"{match.group(1)}-{match.group(2)}"
            # Prioriza a leitura com maior confiança do OCR
            if cand['conf'] > max_conf:
                max_conf = cand['conf']
                best_match = plate
                
    return best_match

def main():
    st.title("🚗 Leitor de Placas de Alta Tolerância")
    yolo_model, ocr_reader = load_models()

    uploaded_file = st.file_uploader("Upload da imagem", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.image(image_rgb, use_column_width=True)
        
        with st.spinner("Analisando cena..."):
            placa = extract_best_plate(image_rgb, yolo_model, ocr_reader)
            
            if placa:
                st.success(f"### Placa Identificada: {placa}")
            else:
                st.error("Não foi possível detectar padrão de placa na imagem.")

if __name__ == '__main__':
    main()
