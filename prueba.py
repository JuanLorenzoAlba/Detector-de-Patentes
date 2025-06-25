from ultralytics import YOLO
import cv2
import numpy as np
import re
from PIL import Image, ImageEnhance
import easyocr
import time
import os
import threading
from queue import Queue
import concurrent.futures

# Inicializar EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Cargar modelo YOLO
model_path = r"C:\Users\Alba Juan\Desktop\PTR\Proyecto Yolo\runs\detect\train\weights\last.pt"
model = YOLO(model_path)

# Crear carpeta para guardar capturas
if not os.path.exists("capturas_patentes"):
    os.makedirs("capturas_patentes")

class DetectorPatentesCaptura:
    def __init__(self):
        self.estado = "BUSCANDO"  # BUSCANDO, CAPTURADA, PROCESANDO, COMPLETADO
        self.contador_detecciones = 0
        self.umbral_detecciones = 5
        self.ultima_bbox = None
        self.imagen_capturada = None
        self.resultado_final = None
        self.progreso_ocr = ""
        
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
        """Verifica si la bbox es estable"""
        if self.ultima_bbox is None:
            self.ultima_bbox = nueva_bbox
            return True
            
        x1, y1, x2, y2 = nueva_bbox
        ux1, uy1, ux2, uy2 = self.ultima_bbox
        
        diff_pos = abs(x1-ux1) + abs(y1-uy1) + abs(x2-ux2) + abs(y2-uy2)
        area_actual = (x2-x1) * (y2-y1)
        area_anterior = (ux2-ux1) * (uy2-uy1)
        
        umbral_movimiento = 20
        umbral_area = 0.2
        
        es_estable = (diff_pos < umbral_movimiento and 
                     abs(area_actual - area_anterior) / area_anterior < umbral_area)
        
        self.ultima_bbox = nueva_bbox
        return es_estable
    
    def capturar_imagen(self, frame, bbox):
        """Captura la imagen de la patente"""
        x1, y1, x2, y2 = bbox
        
        margen = 10
        h, w = frame.shape[:2]
        x1e = max(0, x1 - margen)
        y1e = max(0, y1 - margen)
        x2e = min(w, x2 + margen)
        y2e = min(h, y2 + margen)
        
        patente_crop = frame[y1e:y2e, x1e:x2e]
        
        if patente_crop.size == 0:
            return None
            
        # Evaluar calidad
        gray = cv2.cvtColor(patente_crop, cv2.COLOR_BGR2GRAY)
        nitidez = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        print(f"📊 Nitidez de la captura: {nitidez:.2f}")
        
        if nitidez > 80:  # Umbral más bajo para ser menos estricto
            timestamp = int(time.time())
            filename = f"capturas_patentes/patente_{timestamp}.jpg"
            cv2.imwrite(filename, patente_crop)
            print(f"📸 Imagen capturada y guardada: {filename}")
            return patente_crop
        else:
            print("⚠️ Imagen borrosa, esperando mejor captura...")
            return None
    
    def _mejorar_contraste(self, img):
        """Mejora contraste de forma optimizada"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(img)
    
    def _preprocesar_imagen(self, imagen):
        """Preprocesamiento optimizado"""
        if len(imagen.shape) == 3:
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            gray = imagen
        
        imagenes_procesadas = []
        escalas = [2, 3, 4]  # Volvemos a 3 escalas para mejor precisión
        
        print("🔄 Preprocesando imagen en múltiples escalas...")
        
        for i, escala in enumerate(escalas):
            print(f"   Procesando escala {escala}x... ({i+1}/{len(escalas)})")
            
            # Redimensionar
            resized = cv2.resize(gray, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
            
            # Filtro
            filtered = cv2.bilateralFilter(resized, 9, 75, 75)
            
            # Mejorar contraste
            enhanced = self._mejorar_contraste(filtered)
            
            # Técnicas de umbralización
            # 1. Adaptativa
            thresh1 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            imagenes_procesadas.append(thresh1)
            
            # 2. OTSU
            _, thresh2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            imagenes_procesadas.append(thresh2)
            
            # Guardar imagen procesada para debug
            cv2.imwrite(f"capturas_patentes/procesada_escala_{escala}_adaptativa.jpg", thresh1)
            cv2.imwrite(f"capturas_patentes/procesada_escala_{escala}_otsu.jpg", thresh2)
        
        print(f"✅ Generadas {len(imagenes_procesadas)} variaciones de la imagen")
        return imagenes_procesadas
    
    def _ejecutar_ocr(self, imagenes_procesadas):
        """Ejecuta OCR en todas las imágenes y vota por el mejor resultado"""
        print("🔍 Ejecutando OCR en todas las variaciones...")
        resultados = {}
        
        for i, img in enumerate(imagenes_procesadas):
            print(f"   Analizando imagen {i+1}/{len(imagenes_procesadas)}...")
            
            try:
                # Convertir a RGB
                if len(img.shape) == 2:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # OCR con EasyOCR
                detecciones = reader.readtext(img_rgb, detail=1, width_ths=0.7, height_ths=0.7)
                
                for bbox, texto, conf in detecciones:
                    if conf > 0.3:  # Confianza mínima
                        texto_limpio = ''.join(c for c in texto if c.isalnum()).upper()
                        
                        if len(texto_limpio) >= 5 and "REPUBLICA" not in texto_limpio and "ARGENTINA" not in texto_limpio:
                            es_valida, texto_final = self.validar_patente(texto_limpio)
                            if es_valida:
                                if texto_final not in resultados:
                                    resultados[texto_final] = []
                                resultados[texto_final].append(conf)
                                print(f"      ✓ Candidato encontrado: {texto_final} (confianza: {conf:.2f})")
                                
            except Exception as e:
                print(f"      ⚠️ Error en OCR de imagen {i}: {e}")
                continue
        
        # Calcular puntuación final por patente
        puntuaciones_finales = {}
        for patente, confidencias in resultados.items():
            # Promedio de confianza * número de detecciones
            puntuacion = (sum(confidencias) / len(confidencias)) * len(confidencias)
            puntuaciones_finales[patente] = puntuacion
            print(f"   📊 {patente}: {len(confidencias)} detecciones, puntuación: {puntuacion:.2f}")
        
        if puntuaciones_finales:
            mejor_patente = max(puntuaciones_finales.items(), key=lambda x: x[1])
            print(f"🏆 Mejor resultado: {mejor_patente[0]} (puntuación final: {mejor_patente[1]:.2f})")
            return mejor_patente[0]
        
        return None
    
    def procesar_imagen_capturada(self):
        """Procesa la imagen capturada para extraer la patente"""
        if self.imagen_capturada is None:
            return None
            
        print("\n" + "="*50)
        print("🚀 INICIANDO PROCESAMIENTO DE IMAGEN CAPTURADA")
        print("="*50)
        
        try:
            # Preprocesar imagen
            imagenes_procesadas = self._preprocesar_imagen(self.imagen_capturada)
            
            # Ejecutar OCR
            resultado = self._ejecutar_ocr(imagenes_procesadas)
            
            print("="*50)
            if resultado:
                print(f"✅ PROCESAMIENTO COMPLETADO - PATENTE DETECTADA: {resultado}")
            else:
                print("❌ PROCESAMIENTO COMPLETADO - NO SE PUDO DETECTAR PATENTE")
            print("="*50)
            
            return resultado
            
        except Exception as e:
            print(f"❌ Error durante el procesamiento: {e}")
            return None
    
    def procesar_frame(self, frame):
        """Procesa cada frame del video hasta capturar la imagen"""
        
        if self.estado in ["CAPTURADA", "PROCESANDO", "COMPLETADO"]:
            return frame, True  # Ya capturamos, detener video
        
        # Detectar patentes con YOLO
        results = model(frame, conf=0.1)[0]
        
        patente_detectada = False
        mejor_bbox = None
        mejor_confidence = 0
        
        for box in results.boxes:
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            if confidence > 0.1 and cls_id == 0:
                if confidence > mejor_confidence:
                    mejor_confidence = confidence
                    mejor_bbox = tuple(map(int, box.xyxy[0]))
                    patente_detectada = True
        
        if patente_detectada:
            if self.calcular_estabilidad_bbox(mejor_bbox):
                self.contador_detecciones += 1
                x1, y1, x2, y2 = mejor_bbox
                
                # Mostrar progreso visual
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"Detectando... {self.contador_detecciones}/{self.umbral_detecciones}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Barra de progreso
                progreso = self.contador_detecciones / self.umbral_detecciones
                cv2.rectangle(frame, (10, 10), (300, 30), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (10 + int(290 * progreso), 30), (0, 255, 0), -1)
                cv2.putText(frame, f"Progreso: {int(progreso * 100)}%", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if self.contador_detecciones >= self.umbral_detecciones:
                    imagen_capturada = self.capturar_imagen(frame, mejor_bbox)
                    if imagen_capturada is not None:
                        self.imagen_capturada = imagen_capturada
                        self.estado = "CAPTURADA"
                        print("🎯 ¡IMAGEN CAPTURADA! Cerrando cámara...")
                        return frame, True  # Señal para cerrar cámara
            else:
                self.contador_detecciones = 0
        else:
            self.contador_detecciones = 0
            cv2.putText(frame, "🔍 Buscando patente...", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Posiciona la patente frente a la camara", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, False

def mostrar_resultado_final(resultado, imagen_capturada):
    """Muestra el resultado final en una ventana"""
    # Crear imagen de resultado
    if imagen_capturada is not None:
        # Redimensionar imagen capturada para mostrar
        height, width = imagen_capturada.shape[:2]
        if width > 600:
            scale = 600 / width
            new_width = 600
            new_height = int(height * scale)
            imagen_mostrar = cv2.resize(imagen_capturada, (new_width, new_height))
        else:
            imagen_mostrar = imagen_capturada.copy()
        
        # Crear ventana de resultado
        resultado_img = np.zeros((imagen_mostrar.shape[0] + 150, max(imagen_mostrar.shape[1], 600), 3), dtype=np.uint8)
        
        # Colocar imagen
        resultado_img[100:100+imagen_mostrar.shape[0], :imagen_mostrar.shape[1]] = imagen_mostrar
        
        # Agregar texto del resultado
        if resultado:
            texto_resultado = f"PATENTE DETECTADA: {resultado}"
            color_texto = (0, 255, 0)  # Verde
        else:
            texto_resultado = "NO SE PUDO DETECTAR LA PATENTE"
            color_texto = (0, 0, 255)  # Rojo
        
        cv2.putText(resultado_img, texto_resultado, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color_texto, 2)
        cv2.putText(resultado_img, "Presiona cualquier tecla para salir", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar resultado
        cv2.imshow("RESULTADO - Detector de Patentes", resultado_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ No se pudo mostrar la imagen capturada")

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return
    
    # Configuración de cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    detector = DetectorPatentesCaptura()
    
    print("📷 Detector de patentes iniciado")
    print("🎯 Posiciona la patente frente a la cámara")
    print("📸 La cámara se cerrará automáticamente al capturar")
    print("⌨️ Presiona 'q' para salir antes de capturar")
    
    try:
        # Fase 1: Captura de imagen
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ No se pudo capturar el frame")
                break
            
            frame_procesado, captura_completa = detector.procesar_frame(frame)
            
            cv2.imshow("Detector de Patentes - Capturando", frame_procesado)
            
            # Salir si se presiona 'q' o si se completó la captura
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or captura_completa:
                break
        
        # Cerrar cámara y ventana
        cap.release()
        cv2.destroyAllWindows()
        
        # Fase 2: Procesamiento de imagen capturada
        if detector.imagen_capturada is not None:
            print("\n🔄 Procesando imagen capturada...")
            resultado = detector.procesar_imagen_capturada()
            
            # Fase 3: Mostrar resultado
            print(f"\n🎉 RESULTADO FINAL: {resultado if resultado else 'NO DETECTADA'}")
            mostrar_resultado_final(resultado, detector.imagen_capturada)
        else:
            print("\n❌ No se capturó ninguna imagen")
                
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por usuario")
    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()