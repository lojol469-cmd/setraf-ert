# === LifeModo Multimodal Training Pipeline (Streamlit Version - Advanced & Robust) ===

import os
import time
import json
import gc
import shutil
import fitz
import pytesseract
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import speech_recognition as sr
from gtts import gTTS
import whisper

# === DISABLE GPU for TensorFlow (to avoid conflicts) ===
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

# === CONFIGURATION ===
BASE_DIR = os.path.expanduser("~/lifemodo")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images/train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels/train")
TEXT_DIR = os.path.join(DATASET_DIR, "text")
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
EXPORT_DIR = os.path.join(BASE_DIR, "exported_models")
STATUS_FILE = os.path.join(BASE_DIR, "status.json")

for d in [IMAGES_DIR, LABELS_DIR, TEXT_DIR, AUDIO_DIR, EXPORT_DIR]:
    os.makedirs(d, exist_ok=True)

if os.path.exists(STATUS_FILE):
    with open(STATUS_FILE, "r") as f:
        status = json.load(f)
    if "processed_pdfs" in status and "processed_files" not in status:
        status["processed_files"] = status.pop("processed_pdfs")
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f)
else:
    status = {"processed_files": []}
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)

# === LOG ===
def log(msg):
    st.info(f"[{time.strftime('%H:%M:%S')}] {msg}")

# === STREAMLIT UI ===
st.set_page_config(page_title="LifeModo Multimodal Training", layout="wide")
st.title("🧠 LifeModo Multimodal Training Pipeline (Robuste)")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.header("⚙️ Paramètres")
    PREVIEW_IMAGES = st.checkbox("Prévisualiser les images OCR", value=False, key="preview")
    total_epochs = st.slider("Nombre d'époques YOLO", min_value=1, max_value=200, value=50, key="epochs")
    train_text = st.checkbox("Entraîner sur le texte extrait", value=True)
    train_audio = st.checkbox("Entraîner sur l'audio (si disponible)", value=True)
    export_button = st.button("Exporter les modèles", key="export_btn")

with col2:
    st.header("📊 Statut")
    st.json(status)

# === OCR & ANNOTATION ===
def generate_annotations_with_ocr(image_path, label_output_path):
    image = cv2.imread(image_path)
    if image is None:
        log(f"Image corrompue ignorée : {image_path}")
        return False

    h, w, _ = image.shape
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception as e:
        log(f"Erreur OCR sur {image_path} : {e}")
        return False

    found = False
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if not text:
            continue
        x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        bw_norm, bh_norm = bw / w, bh / h
        with open(label_output_path, "a") as label_file:
            label_file.write(f"0 {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")
        found = True
        if PREVIEW_IMAGES:
            cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    # Détection d'objets avec YOLO pré-entraîné pour ajouter des annotations d'objets
    try:
        object_model = YOLO('yolov8n.pt')
        results = object_model(image_path)
        object_found = False
        for r in results:
            for box in r.boxes:
                if box.conf > 0.5:
                    x_center, y_center, bw_norm, bh_norm = box.xywhn[0].tolist()
                    with open(label_output_path, "a") as label_file:
                        label_file.write(f"4 {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")
                    object_found = True
                    if PREVIEW_IMAGES:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        if object_found:
            log(f"Objets détectés dans : {image_path}")
    except Exception as e:
        log(f"Erreur détection objets sur {image_path} : {e}")

    if found and PREVIEW_IMAGES:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=os.path.basename(image_path))

    log(f"Annotations OCR générées pour : {image_path} - Trouvé : {found}")
    return found

# === EXTRACTION TEXTE PDF ===
def extract_text_from_pdf(pdf_path, text_output_path):
    pdf = fitz.open(pdf_path)
    text = ""
    for page in pdf:
        text += page.get_text()
    with open(text_output_path, "w") as f:
        f.write(text)
    log(f"Texte extrait pour : {pdf_path}")
    pdf.close()

# === AUDIO ===
def process_audio(audio_path, text_output_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        with open(text_output_path, "w") as f:
            f.write(result["text"])
        log(f"Audio transcrit pour : {audio_path}")
    except Exception as e:
        log(f"Erreur transcription audio : {e}")

# === EXTRACTION DES LÉGENDES PROCHES DES IMAGES ===
def extract_captions_near_images(pdf, page, page_num, pdf_name):
    try:
        blocks = page.get_text("blocks")
        images = page.get_images(full=True)
        captions = []

        for img in images:
            xref = img[0]
            bbox = page.get_image_bbox(xref)
            for b in blocks:
                bx0, by0, bx1, by1, text, *_ = b
                if abs(by0 - bbox.y1) < 80 or abs(bbox.y0 - by1) < 80:
                    if len(text.strip()) > 5:
                        captions.append(text.strip())

        if captions:
            text_output_path = os.path.join(TEXT_DIR, f"{pdf_name}_page{page_num+1}_captions.txt")
            with open(text_output_path, "w") as f:
                for caption in captions:
                    f.write(caption + "\n")
            log(f"Légendes extraites page {page_num+1} : {len(captions)} trouvées.")

            for i, caption in enumerate(captions):
                img = np.zeros((100, 800, 3), dtype=np.uint8) * 255
                cv2.putText(img, caption, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                caption_filename = f"{pdf_name}_page{page_num+1}_caption_{i}.png"
                full_caption_path = os.path.join(IMAGES_DIR, caption_filename)
                cv2.imwrite(full_caption_path, img)
                label_caption_path = os.path.join(LABELS_DIR, f"{pdf_name}_page{page_num+1}_caption_{i}.txt")
                with open(label_caption_path, "w") as f:
                    f.write("3 0.500000 0.500000 1.000000 1.000000\n")
                generate_annotations_with_ocr(full_caption_path, label_caption_path)
                if PREVIEW_IMAGES:
                    h, w, _ = img.shape
                    cv2.rectangle(img, (0, 0), (w-1, h-1), (255, 0, 0), 2)
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=caption_filename)
    except Exception as e:
        log(f"Erreur extraction légendes page {page_num+1} : {e}")

# === EXTRACTION DE LA CARTE D'IMAGES ===
def extract_image_map(pdf, page, page_num, pdf_name):
    try:
        images = page.get_images(full=True)
        map_data = []
        for img in images:
            xref = img[0]
            bbox = page.get_image_bbox(xref)
            map_data.append({
                "xref": xref,
                "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                "page": page_num + 1
            })
        map_path = os.path.join(TEXT_DIR, f"{pdf_name}_page{page_num+1}_map.json")
        with open(map_path, "w") as f:
            json.dump(map_data, f, indent=2)
        log(f"Carte d’images générée page {page_num+1}.")

        if map_data:
            page_rect = page.rect
            w = int(page_rect.width)
            h = int(page_rect.height)
            img = np.zeros((h, w, 3), dtype=np.uint8) * 255
            for bbox in map_data:
                cv2.rectangle(img, (int(bbox['bbox'][0]), int(bbox['bbox'][1])), (int(bbox['bbox'][2]), int(bbox['bbox'][3])), (0, 255, 0), 2)
            map_filename = f"{pdf_name}_page{page_num+1}_map.png"
            full_map_path = os.path.join(IMAGES_DIR, map_filename)
            cv2.imwrite(full_map_path, img)
            label_output_path = os.path.join(LABELS_DIR, f"{pdf_name}_page{page_num+1}_map.txt")
            with open(label_output_path, "w") as f:
                f.write("2 0.500000 0.500000 1.000000 1.000000\n")
            generate_annotations_with_ocr(full_map_path, label_output_path)
            if PREVIEW_IMAGES:
                h_img, w_img, _ = img.shape
                cv2.rectangle(img, (0, 0), (w_img-1, h_img-1), (255, 0, 0), 2)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=map_filename)
    except Exception as e:
        log(f"Erreur génération carte d’images : {e}")

# === EXTRACTION DES CROQUIS / CARTES VECTORIELLES ===
def extract_drawings(pdf, page, page_num, pdf_name):
    drawings = page.get_drawings()
    if drawings:
        pix = page.get_pixmap()
        drawing_filename = f"{pdf_name}_page{page_num+1}_drawings.png"
        full_drawing_path = os.path.join(IMAGES_DIR, drawing_filename)
        pix.save(full_drawing_path)
        label_output_path = os.path.join(LABELS_DIR, f"{pdf_name}_page{page_num+1}_drawings.txt")
        with open(label_output_path, "w") as label_file:
            label_file.write("1 0.500000 0.500000 1.000000 1.000000\n")
        generate_annotations_with_ocr(full_drawing_path, label_output_path)
        if PREVIEW_IMAGES:
            ann_image = cv2.imread(full_drawing_path)
            h, w, _ = ann_image.shape
            cv2.rectangle(ann_image, (0, 0), (w-1, h-1), (255, 0, 0), 2)
            st.image(cv2.cvtColor(ann_image, cv2.COLOR_BGR2RGB), caption=drawing_filename)
        log(f"Croquis / cartes / graphiques / illustrations enregistrés page {page_num+1}.")

# === EXTRACTION PDF COMPLÈTE ===
def extract_from_pdf(pdf_path, pdf_name):
    if pdf_name in status["processed_files"]:
        log(f"{pdf_name} déjà traité.")
        return

    log(f"Début extraction PDF : {pdf_name}")
    pdf = fitz.open(pdf_path)
    total_images = sum(len(pdf[p].get_images(full=True)) for p in range(len(pdf)))
    progress_bar = st.progress(0)
    processed = 0

    text_output_path = os.path.join(TEXT_DIR, f"{pdf_name}.txt")
    extract_text_from_pdf(pdf_path, text_output_path)

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        extract_captions_near_images(pdf, page, page_num, pdf_name)
        extract_image_map(pdf, page, page_num, pdf_name)
        extract_drawings(pdf, page, page_num, pdf_name)

        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            image_filename = f"{pdf_name}_page{page_num+1}_{img_index}.png"
            full_image_path = os.path.join(IMAGES_DIR, image_filename)

            if not os.path.exists(full_image_path):
                with open(full_image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                label_output_path = os.path.join(LABELS_DIR, f"{pdf_name}_page{page_num+1}_{img_index}.txt")
                generate_annotations_with_ocr(full_image_path, label_output_path)

            processed += 1
            progress_bar.progress(processed / max(total_images, 1))

    pdf.close()
    status["processed_files"].append(pdf_name)
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)
    log(f"Extraction terminée pour : {pdf_name}")

# === ENTRAÎNEMENT VISUEL ===
def train_yolo(pdf_name, total_epochs=50):
    yaml_path = os.path.join(BASE_DIR, f"data_{pdf_name}.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""
path: {BASE_DIR}/dataset
train: images/train
val: images/train
nc: 5
names: ['texte', 'drawing', 'map', 'legend', 'object']
""")

    model_dir = os.path.join(BASE_DIR, f"{pdf_name}_model")
    weights_dir = os.path.join(model_dir, "weights")
    last_checkpoint = os.path.join(weights_dir, "last.pt")
    os.makedirs(weights_dir, exist_ok=True)

    if os.path.exists(last_checkpoint):
        model = YOLO(last_checkpoint)
        log("Checkpoint trouvé. Reprise de l'entraînement.")
        resume = True
    else:
        model = YOLO('yolov8n.pt')
        log("Aucun checkpoint trouvé. Nouveau modèle YOLO.")
        resume = False

    start_time = time.time()
    model.train(
        data=yaml_path,
        epochs=total_epochs,
        imgsz=640,
        project=BASE_DIR,
        name=f"{pdf_name}_model",
        batch=16,
        resume=resume
    )
    elapsed = time.time() - start_time
    log(f"Entraînement YOLO terminé pour {pdf_name} ({elapsed/60:.2f} min).")
    return model

# === ENTRAÎNEMENT TEXTE ===
def train_text_model(pdf_name):
    text_path = os.path.join(TEXT_DIR, f"{pdf_name}.txt")
    if not os.path.exists(text_path):
        log(f"Aucun texte trouvé pour {pdf_name}")
        return
    with open(text_path, "r") as f:
        text = f.read()
    log(f"Entraînement texte pour {pdf_name} terminé (placeholder NLP).")

# === ENTRAÎNEMENT AUDIO ===
def train_audio_model(pdf_name, audio_path=None):
    if audio_path is None:
        log("Aucun fichier audio fourni.")
        return
    text_output_path = os.path.join(TEXT_DIR, f"{pdf_name}_audio.txt")
    process_audio(audio_path, text_output_path)
    log(f"Entraînement audio pour {pdf_name} terminé (placeholder).")

# === EXPORTATION DES MODÈLES ===
def export_model_formats(model, pdf_name):
    model_path = os.path.join(BASE_DIR, f"{pdf_name}_model/weights/best.pt")
    if model is None:
        model = YOLO(model_path)

    log(f"Export des modèles pour {pdf_name}...")
    onnx_default = model.export(format="onnx")
    onnx_target = os.path.join(EXPORT_DIR, f"{pdf_name}.onnx")
    shutil.move(onnx_default, onnx_target)

    saved_default = model.export(format="saved_model")
    saved_target = os.path.join(EXPORT_DIR, f"{pdf_name}_tf")
    shutil.move(saved_default, saved_target)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_target)
    tflite_model = converter.convert()
    with open(os.path.join(EXPORT_DIR, f"{pdf_name}.tflite"), "wb") as f:
        f.write(tflite_model)

    os.system("pip install -q tensorflowjs")
    tfjs_dir = os.path.join(EXPORT_DIR, f"{pdf_name}_tfjs")
    if os.path.exists(tfjs_dir):
        shutil.rmtree(tfjs_dir)
    os.makedirs(tfjs_dir, exist_ok=True)
    os.system(f"tensorflowjs_converter --input_format=tf_saved_model {saved_target} {tfjs_dir}")
    log(f"✅ Export terminé pour {pdf_name}")

# === UPLOAD ===
st.header("📤 Téléversez vos fichiers")
uploaded_pdfs = st.file_uploader("Téléversez vos PDFs", type=["pdf"], accept_multiple_files=True)
uploaded_audios = st.file_uploader("Téléversez vos Audios (optionnel)", type=["wav", "mp3"], accept_multiple_files=True)

if uploaded_pdfs:
    for uploaded_pdf in uploaded_pdfs:
        pdf_path = os.path.join(BASE_DIR, uploaded_pdf.name)
        pdf_name = os.path.splitext(uploaded_pdf.name)[0]
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
        extract_from_pdf(pdf_path, pdf_name)

if uploaded_audios:
    for uploaded_audio in uploaded_audios:
        audio_path = os.path.join(AUDIO_DIR, uploaded_audio.name)
        pdf_name = os.path.splitext(uploaded_audio.name)[0]
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio.read())

# === ENTRAÎNEMENT MANUEL ===
if st.button("🏋️ Entraîner Modèles"):
    for pdf_name in status["processed_files"]:
        model = train_yolo(pdf_name, total_epochs)
        if train_text:
            train_text_model(pdf_name)
        if train_audio:
            audio_path = os.path.join(AUDIO_DIR, f"{pdf_name}.wav")
            if os.path.exists(audio_path):
                train_audio_model(pdf_name, audio_path)
        export_model_formats(model, pdf_name)

# === EXPORT MANUEL ===
if export_button:
    for pdf_name in status["processed_files"]:
        export_model_formats(None, pdf_name)
    st.success("✅ Export terminé pour tous les modèles.")