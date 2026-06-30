import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Layout mais largo para acomodar o menu lateral
st.set_page_config(page_title="YOLOv8 Avançado", layout="wide")

st.title("👁️ Detecção Avançada com YOLOv8")
st.write("Ajuste a inteligência e a visão do modelo no menu lateral.")

# ----------------- MENU LATERAL -----------------
st.sidebar.header("⚙️ Parâmetros do Modelo")

# 1. Tamanho do Modelo (Permite baixar modelos mais inteligentes)
escolha_modelo = st.sidebar.selectbox(
    "1. Inteligência da IA (Tamanho)",
    [
        "yolov8n.pt (Nano - Muito Rápido, Menos Preciso)", 
        "yolov8s.pt (Small - Equilibrado)", 
        "yolov8m.pt (Medium - Mais Lento na CPU, Muito Preciso)"
    ]
)
arquivo_modelo = escolha_modelo.split(" ")[0] # Extrai apenas o "yolov8m.pt"

# 2. Confiança
confianca = st.sidebar.slider(
    "2. Limite de Confiança (conf)", 
    min_value=0.01, max_value=1.0, value=0.25, step=0.01,
    help="Valores menores mostram mais objetos, mas podem gerar falsos positivos."
)

# 3. Resolução da Imagem (Lente de aumento)
tamanho_img = st.sidebar.select_slider(
    "3. Resolução de Inferência (imgsz)",
    options=[640, 736, 800, 1024, 1280],
    value=1024,
    help="Aumente para 1024 ou 1280 para achar objetos minúsculos na pintura."
)

# ----------------- LÓGICA PRINCIPAL -----------------
# Cache adaptável: recarrega apenas se o usuário mudar de Nano para Medium, por exemplo
@st.cache_resource
def load_model(nome_modelo):
    return YOLO(nome_modelo)

model = load_model(arquivo_modelo)

uploaded_file = st.file_uploader("Escolha a imagem (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tratamento de imagem robusto (forçando RGB para evitar erro de 4 canais)
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Imagem Original", use_column_width=True)

    with col2:
        if st.button("Detectar Objetos", type="primary"):
            with st.spinner(f"Processando com {arquivo_modelo}..."):
                
                # A MÁGICA: Passamos a confiança e o tamanho da imagem para a IA!
                results = model(img_array, conf=confianca, imgsz=tamanho_img)
                
                annotated_image = results[0].plot()
                
                st.image(annotated_image, caption="Resultado Avançado", use_column_width=True)
                
                total_objetos = len(results[0].boxes)
                st.success(f"Encontrados {total_objetos} objeto(s).")
                
                if total_objetos > 0:
                    nomes = [results[0].names[int(classe)] for classe in results[0].boxes.cls]
                    st.write("**Classes identificadas:**", ", ".join(set(nomes)))