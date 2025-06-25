from ultralytics import YOLO
import cv2
import numpy as np
import re
from PIL import Image, ImageEnhance
import easyocr
import time
import os

# Inicializar EasyOCR
reader = easyocr.Reader(['en'], gpu=True)

# Cargar modelo YOLO
model_path = r"C:\Users\Alba Juan\Desktop\PTR\Proyecto Yolo\runs\detect\train\weights\last.pt"
model = YOLO(model_path)

# Crear carpeta para guardar capturas
if not os.path.exists("capturas_patentes"):
    os.makedirs("capturas_patentes")

class DetectorPatentes:
    def __init__(self):
        self.estado = "BUSCANDO"  # BUSCANDO, DETECTADA, PROCESANDO, COMPLETADO
        self.contador_detecciones = 0
        self.umbral_detecciones = 5  # Detecciones consecutivas antes de capturar
        self.ultima_bbox = None
        self.imagen_capturada = None
        self.resultado_final = None
        self.tiempo_resultado = 0  # Timestamp cuando se completa la detección
        self.duracion_mostrar = 3.0  # Segundos para mostrar el resultado
        
    def validar_patente(self, texto):
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
    
    def calcular_estabilidad_bbox(self, nueva_bbox):
        """Verifica si la bbox es estable (no se mueve mucho)"""
        if self.ultima_bbox is None:
            self.ultima_bbox = nueva_bbox
            return True
            
        x1, y1, x2, y2 = nueva_bbox
        ux1, uy1, ux2, uy2 = self.ultima_bbox
        
        # Calcular diferencia en posición y tamaño
        diff_pos = abs(x1-ux1) + abs(y1-uy1) + abs(x2-ux2) + abs(y2-uy2)
        area_actual = (x2-x1) * (y2-y1)
        area_anterior = (ux2-ux1) * (uy2-uy1)
        
        # Si la diferencia es pequeña, consideramos que es estable
        umbral_movimiento = 20
        umbral_area = 0.2  # 20% de diferencia máxima en área
        
        es_estable = (diff_pos < umbral_movimiento and 
                     abs(area_actual - area_anterior) / area_anterior < umbral_area)
        
        self.ultima_bbox = nueva_bbox
        return es_estable
    
    def capturar_mejor_frame(self, frame, bbox):
        """Captura y guarda el mejor frame de la patente"""
        x1, y1, x2, y2 = bbox
        
        # Expandir bbox ligeramente para mejor contexto
        margen = 10
        h, w = frame.shape[:2]
        x1e = max(0, x1 - margen)
        y1e = max(0, y1 - margen)
        x2e = min(w, x2 + margen)
        y2e = min(h, y2 + margen)
        
        # Extraer región de la patente
        patente_crop = frame[y1e:y2e, x1e:x2e]
        
        if patente_crop.size == 0:
            return None
            
        # Evaluar calidad de la imagen (nitidez)
        gray = cv2.cvtColor(patente_crop, cv2.COLOR_BGR2GRAY)
        nitidez = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        print(f"📊 Nitidez de la imagen: {nitidez:.2f}")
        
        # Solo capturar si tiene buena nitidez
        if nitidez > 100:  # Umbral de nitidez mínima
            timestamp = int(time.time())
            filename = f"capturas_patentes/patente_{timestamp}.jpg"
            cv2.imwrite(filename, patente_crop)
            print(f"📸 Imagen capturada: {filename}")
            return patente_crop
        else:
            print("⚠️ Imagen muy borrosa, esperando mejor captura...")
            return None
    
    def mejorar_contraste(self, img):
        """Mejora el contraste de la imagen"""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img)
        pil_img = Image.fromarray(img_clahe)
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(2.0)
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
        sharpened = sharpness_enhancer.enhance(2.0)
        return np.array(sharpened)
    
    def preprocesar_imagen_capturada(self, imagen):
        """Preprocesa la imagen capturada con múltiples técnicas"""
        if len(imagen.shape) == 3:
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            gray = imagen
        
        imagenes_procesadas = []
        
        # Múltiples escalas y procesamientos
        escalas = [2, 3, 4]
        
        for escala in escalas:
            # Redimensionar
            resized = cv2.resize(gray, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
            
            # Aplicar filtros
            filtered = cv2.bilateralFilter(resized, 11, 17, 17)
            
            # Mejorar contraste
            enhanced = self.mejorar_contraste(filtered)
            
            # Diferentes técnicas de umbralización
            # 1. Umbralización binaria simple
            _, thresh1 = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
            imagenes_procesadas.append(thresh1)
            
            # 2. Umbralización adaptativa
            thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            imagenes_procesadas.append(thresh2)
            
            # 3. Umbralización OTSU
            _, thresh3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            imagenes_procesadas.append(thresh3)
        
        return imagenes_procesadas
    
    def procesar_ocr_multiple(self, imagenes_procesadas):
        """Ejecuta OCR en múltiples versiones de la imagen y vota por el mejor resultado"""
        resultados = {}
        
        for i, img in enumerate(imagenes_procesadas):
            # Convertir a RGB para EasyOCR
            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Guardar imagen procesada para debug
            cv2.imwrite(f"capturas_patentes/procesada_{i}.jpg", img)
            
            # OCR con EasyOCR
            try:
                detecciones = reader.readtext(img_rgb, detail=1)
                
                for bbox, texto, conf in detecciones:
                    if conf > 0.3:  # Confianza mínima
                        texto_limpio = ''.join(c for c in texto if c.isalnum()).upper()
                        
                        # Filtrar textos muy cortos o que contengan palabras comunes
                        if len(texto_limpio) < 5 or "REPUBLICA" in texto_limpio or "ARGENTINA" in texto_limpio:
                            continue
                        
                        es_valida, texto_final = self.validar_patente(texto_limpio)
                        if es_valida:
                            if texto_final not in resultados:
                                resultados[texto_final] = 0
                            resultados[texto_final] += conf
                            print(f"🔍 Candidato: {texto_final} (conf: {conf:.2f})")
                
            except Exception as e:
                print(f"⚠️ Error en OCR para imagen {i}: {e}")
                continue
        
        # Retornar el resultado con mayor puntuación acumulada
        if resultados:
            mejor_resultado = max(resultados.items(), key=lambda x: x[1])
            print(f"🏆 Mejor resultado: {mejor_resultado[0]} (puntuación: {mejor_resultado[1]:.2f})")
            return mejor_resultado[0]
        
        return None
    
    def reiniciar_para_nueva_busqueda(self):
        """Reinicia el detector para buscar una nueva patente"""
        self.estado = "BUSCANDO"
        self.contador_detecciones = 0
        self.ultima_bbox = None
        self.imagen_capturada = None
        self.resultado_final = None
        self.tiempo_resultado = 0
        print("🔄 Buscando nueva patente...")
    
    def procesar_frame(self, frame):
        """Procesa un frame del video según el estado actual"""
        
        # Estado: COMPLETADO - Mostrar resultado y luego reiniciar automáticamente
        if self.estado == "COMPLETADO":
            tiempo_transcurrido = time.time() - self.tiempo_resultado
            tiempo_restante = self.duracion_mostrar - tiempo_transcurrido
            
            if tiempo_restante > 0:
                # Mostrar resultado con countdown
                cv2.putText(frame, f"PATENTE: {self.resultado_final}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"Nueva busqueda en: {tiempo_restante:.1f}s", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                return frame, False
            else:
                # Tiempo terminado, reiniciar automáticamente
                self.reiniciar_para_nueva_busqueda()
                return frame, False
        
        # Detectar patentes con YOLO
        results = model(frame, conf=0.1)[0]
        
        patente_detectada = False
        mejor_bbox = None
        mejor_confidence = 0
        
        # Buscar la mejor detección
        for box in results.boxes:
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            if confidence > 0.1 and cls_id == 0:
                if confidence > mejor_confidence:
                    mejor_confidence = confidence
                    mejor_bbox = tuple(map(int, box.xyxy[0]))
                    patente_detectada = True
        
        # Estado: BUSCANDO
        if self.estado == "BUSCANDO":
            if patente_detectada:
                # Verificar estabilidad de la detección
                if self.calcular_estabilidad_bbox(mejor_bbox):
                    self.contador_detecciones += 1
                    x1, y1, x2, y2 = mejor_bbox
                    
                    # Dibujar bbox en verde con contador
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Detectando... {self.contador_detecciones}/{self.umbral_detecciones}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Si tenemos suficientes detecciones estables, capturar
                    if self.contador_detecciones >= self.umbral_detecciones:
                        imagen_capturada = self.capturar_mejor_frame(frame, mejor_bbox)
                        if imagen_capturada is not None:
                            self.imagen_capturada = imagen_capturada
                            self.estado = "PROCESANDO"
                            print("🎯 Patente detectada estable. Iniciando procesamiento OCR...")
                else:
                    # Reiniciar contador si la detección no es estable
                    self.contador_detecciones = 0
            else:
                # No hay detección, reiniciar contador
                self.contador_detecciones = 0
                cv2.putText(frame, "Buscando patente...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Estado: PROCESANDO
        elif self.estado == "PROCESANDO":
            cv2.putText(frame, "Procesando OCR...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Procesar la imagen capturada
            imagenes_procesadas = self.preprocesar_imagen_capturada(self.imagen_capturada)
            resultado = self.procesar_ocr_multiple(imagenes_procesadas)
            
            if resultado:
                self.resultado_final = resultado
                self.estado = "COMPLETADO"
                self.tiempo_resultado = time.time()  # Marcar el tiempo cuando se completa
                print(f"✅ PATENTE DETECTADA: {resultado}")
            else:
                # Si no se pudo leer, volver a buscar
                print("❌ No se pudo leer la patente. Reintentando...")
                self.estado = "BUSCANDO"
                self.contador_detecciones = 0
                self.imagen_capturada = None
        
        return frame, False

def main():
    # Configurar cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = DetectorPatentes()
    
    print("📷 Detector de patentes iniciado")
    print("🎯 Posiciona la patente frente a la cámara")
    print("🔄 El detector buscará automáticamente nuevas patentes")
    print("⌨️ Presiona 'q' para salir")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No se pudo capturar el frame")
            break
        
        # Procesar frame
        frame_procesado, terminado = detector.procesar_frame(frame)
        
        # Mostrar frame
        cv2.imshow("Detector de Patentes - Búsqueda Continua", frame_procesado)
        
        # Control de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()