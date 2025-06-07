from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np

# Ruta al modelo YOLOv8 entrenado
model_path = r"C:\Users\Alba Juan\Desktop\PTR\Proyecto Yolo\runs\detect\train\weights\last.pt"
model = YOLO(model_path)

# Ruta a la imagen de prueba
img_path = r"C:\Users\Alba Juan\Desktop\PTR\Proyecto Yolo\datasets\images\train\patente40.JPG"
image = cv2.imread(img_path)

if image is None:
    print(f"⚠️ No se pudo cargar la imagen: {img_path}")
    exit()

# 🧠 OCR inteligente con múltiples configuraciones
def ocr_inteligente(thresh_img):
    configs = [
        "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ]
    for i, config in enumerate(configs):
        text = pytesseract.image_to_string(thresh_img, config=config).strip()
        print(f"🧪 Intento OCR #{i+1} (psm {config.split()[1]}): '{text}'")
        if len(text.replace(" ", "")) >= 5:  # mínimo 5 caracteres útiles
            return text
    return ""

# 🔎 Detección con YOLO
results = model(image, conf=0.1)[0]
results.show()

if not results.boxes or len(results.boxes) == 0:
    print("⚠️ No se detectaron objetos")
    cv2.imshow("Imagen", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# 📦 Iterar sobre detecciones
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = float(box.conf[0])
    cls_id = int(box.cls[0])

    print(f"🔎 Detección -> Clase: {cls_id}, Confianza: {confidence:.2f}, BBox: ({x1},{y1}) a ({x2},{y2})")

    if confidence > 0.1 and cls_id == 0:
        patente_crop = image[y1:y2, x1:x2]

        # 🧼 Preprocesamiento para OCR
        gray = cv2.cvtColor(patente_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        gray = cv2.equalizeHist(gray)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 🔧 Morfología
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        # 👁 Mostrar recorte enviado a OCR
        cv2.imshow("Recorte para OCR", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 🔠 Ejecutar OCR
        text = ocr_inteligente(thresh)

        if text:
            print("✅ Texto detectado:", text)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        else:
            print("🧐 OCR no detectó texto")

        # 📦 Dibujar caja
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 🖼 Mostrar imagen final
cv2.imshow("YOLO + OCR", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
