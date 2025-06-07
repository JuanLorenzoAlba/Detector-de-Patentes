from ultralytics import YOLO
import cv2
import numpy as np
import re
from PIL import Image, ImageEnhance
import easyocr

# Inicializar EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Cargar modelo YOLO
model_path = r"C:\Users\Alba Juan\Desktop\PTR\Proyecto Yolo\runs\detect\train\weights\last.pt"
model = YOLO(model_path)

# Validar texto tipo patente
def validar_patente(texto):
    texto = texto.replace(" ", "").replace("-", "").upper()
    patrones = [
        r'^[A-Z]{3}\d{3}$',        # ABC123
        r'^[A-Z]{2}\d{3}[A-Z]{2}$',# AB123CD
        r'^[A-Z]\d{3}[A-Z]{3}$',   # A123BCD
        r'^\d{3}[A-Z]{3}$',        # 123ABC
    ]
    for patron in patrones:
        if re.match(patron, texto):
            return True, texto
    if 6 <= len(texto) <= 7 and texto.isalnum():
        return True, texto
    return False, texto

# Mejora de contraste
def mejorar_contraste(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    pil_img = Image.fromarray(img_clahe)
    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced = enhancer.enhance(2.0)
    sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
    sharpened = sharpness_enhancer.enhance(2.0)
    return np.array(sharpened)

# Preprocesamiento manual
def preprocesar_manual(patente_crop):
    if len(patente_crop.shape) == 3:
        gray = cv2.cvtColor(patente_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = patente_crop
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = mejorar_contraste(gray)
    _, manual_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return manual_thresh

# OCR solo en imagen procesada manualmente
def ocr_easyocr_manual(img_proc):
    if len(img_proc.shape) == 2:
        img_rgb = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
    
    detecciones = reader.readtext(img_rgb)
    for bbox, texto, conf in detecciones:
        texto_limpio = ''.join(c for c in texto if c.isalnum()).upper()
        if len(texto_limpio) < 5 or "REPUBLICA" in texto_limpio or "ARGENTINA" in texto_limpio:
            continue
        es_valida, texto_final = validar_patente(texto_limpio)
        if es_valida:
            return texto_final
    return ""

# Activar cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()

print("📷 Cámara iniciada. Presioná 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo capturar el frame")
        break

    results = model(frame, conf=0.1)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        cls_id = int(box.cls[0])

        if confidence > 0.1 and cls_id == 0:
            margen = 5
            x1e = max(0, x1 - margen)
            y1e = max(0, y1 - margen)
            x2e = min(frame.shape[1], x2 + margen)
            y2e = min(frame.shape[0], y2 + margen)

            crop = frame[y1e:y2e, x1e:x2e]

            if crop.size == 0:
                continue

            img_proc = preprocesar_manual(crop)
            texto_detectado = ocr_easyocr_manual(img_proc)

            if texto_detectado:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, texto_detectado, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("Detector de Patentes", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
