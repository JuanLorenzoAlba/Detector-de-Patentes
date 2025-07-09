from ultralytics import YOLO
import cv2
import numpy as np
import re
from PIL import Image, ImageEnhance
import easyocr
import time
import os
import json

# Inicializar EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Cargar modelo YOLO
model_path = r"C:\ProyectosVarios\Detector de patentes\Detector-de-patentes\runs\detect\train\weights\last.pt"
model = YOLO(model_path)

# Crear carpeta para guardar capturas
if not os.path.exists("capturas_patentes"):
    os.makedirs("capturas_patentes")

class BaseDatosMultas:
    def __init__(self, archivo_db="patentes_multadas.json"):
        self.archivo_db = archivo_db
        self.patentes_multadas = self.cargar_base_datos()
    
    def cargar_base_datos(self):
        """Carga la base de datos de patentes multadas desde un archivo JSON"""
        try:
            if os.path.exists(self.archivo_db):
                with open(self.archivo_db, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Crear archivo con datos de ejemplo si no existe
                datos_ejemplo = {
                    "AB123CD": {
                        "multas": [
                            {"fecha": "2024-01-15", "tipo": "Exceso de velocidad", "monto": 15000},
                            {"fecha": "2024-02-20", "tipo": "Estacionamiento prohibido", "monto": 8000}
                        ],
                        "total_multas": 2,
                        "monto_total": 23000,
                        "estado": "PENDIENTE"
                    },
                    "XYZ789": {
                        "multas": [
                            {"fecha": "2024-03-10", "tipo": "Semáforo en rojo", "monto": 25000}
                        ],
                        "total_multas": 1,
                        "monto_total": 25000,
                        "estado": "PENDIENTE"
                    },
                    "DEF456": {
                        "multas": [
                            {"fecha": "2024-01-05", "tipo": "Zona de carga", "monto": 12000}
                        ],
                        "total_multas": 1,
                        "monto_total": 12000,
                        "estado": "PAGADA"
                    }
                }
                self.guardar_base_datos(datos_ejemplo)
                print(f"📋 Base de datos creada con ejemplos en: {self.archivo_db}")
                return datos_ejemplo
        except Exception as e:
            print(f"⚠️ Error al cargar base de datos: {e}")
            return {}
    
    def guardar_base_datos(self, datos):
        """Guarda la base de datos en el archivo JSON"""
        try:
            with open(self.archivo_db, 'w', encoding='utf-8') as f:
                json.dump(datos, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Error al guardar base de datos: {e}")
    
    def consultar_patente(self, patente):
        """Consulta si una patente tiene multas"""
        patente_limpia = patente.replace(" ", "").replace("-", "").upper()
        
        if patente_limpia in self.patentes_multadas:
            info = self.patentes_multadas[patente_limpia]
            return {
                "tiene_multas": True,
                "info": info
            }
        else:
            return {
                "tiene_multas": False,
                "info": None
            }
    
    def agregar_patente_multada(self, patente, tipo_multa, monto, fecha=None):
        """Agrega una nueva multa a la base de datos"""
        if fecha is None:
            fecha = time.strftime("%Y-%m-%d")
        
        patente_limpia = patente.replace(" ", "").replace("-", "").upper()
        
        if patente_limpia not in self.patentes_multadas:
            self.patentes_multadas[patente_limpia] = {
                "multas": [],
                "total_multas": 0,
                "monto_total": 0,
                "estado": "PENDIENTE"
            }
        
        nueva_multa = {
            "fecha": fecha,
            "tipo": tipo_multa,
            "monto": monto
        }
        
        self.patentes_multadas[patente_limpia]["multas"].append(nueva_multa)
        self.patentes_multadas[patente_limpia]["total_multas"] += 1
        self.patentes_multadas[patente_limpia]["monto_total"] += monto
        
        self.guardar_base_datos(self.patentes_multadas)
        print(f"✅ Multa agregada para patente {patente_limpia}")

class DetectorPatentes:
    def __init__(self):
        self.estado = "BUSCANDO"  # BUSCANDO, DETECTADA, PROCESANDO, COMPLETADO
        self.contador_detecciones = 0
        self.umbral_detecciones = 5  # Detecciones consecutivas antes de capturar
        self.ultima_bbox = None
        self.imagen_capturada = None
        self.resultado_final = None
        self.tiempo_resultado = 0  # Timestamp cuando se completa la detección
        self.duracion_mostrar = 5.0  # Segundos para mostrar el resultado (aumentado para mostrar info de multas)
        
        # Inicializar base de datos de multas
        self.db_multas = BaseDatosMultas()
        self.info_multa = None
        
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
            
            # Filtrado bilateral para reducción de ruido
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
    
    def consultar_multas(self, patente):
        """Consulta si la patente tiene multas y guarda la información"""
        resultado = self.db_multas.consultar_patente(patente)
        self.info_multa = resultado
        
        if resultado["tiene_multas"]:
            info = resultado["info"]
            print(f"🚨 PATENTE CON MULTAS: {patente}")
            print(f"   Total multas: {info['total_multas']}")
            print(f"   Monto total: ${info['monto_total']:,}")
            print(f"   Estado: {info['estado']}")
            
            for i, multa in enumerate(info["multas"], 1):
                print(f"   Multa {i}: {multa['tipo']} - ${multa['monto']:,} ({multa['fecha']})")
        else:
            print(f"✅ PATENTE LIMPIA: {patente} - Sin multas registradas")
    
    def reiniciar_para_nueva_busqueda(self):
        """Reinicia el detector para buscar una nueva patente"""
        self.estado = "BUSCANDO"
        self.contador_detecciones = 0
        self.ultima_bbox = None
        self.imagen_capturada = None
        self.resultado_final = None
        self.tiempo_resultado = 0
        self.info_multa = None
        print("🔄 Buscando nueva patente...")
    
    def dibujar_info_multas(self, frame):
        """Dibuja la información de multas en el frame"""
        if not self.info_multa:
            return frame
        
        y_pos = 60
        line_height = 35
        
        # Dibujar patente detectada
        cv2.putText(frame, f"PATENTE: {self.resultado_final}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        y_pos += line_height
        
        if self.info_multa["tiene_multas"]:
            info = self.info_multa["info"]
            
            # Estado de multa (color rojo para multas pendientes)
            color = (0, 0, 255) if info["estado"] == "PENDIENTE" else (0, 165, 255)  # Rojo o naranja
            cv2.putText(frame, f"MULTADA - {info['estado']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            y_pos += line_height
            
            # Información de multas
            cv2.putText(frame, f"Multas: {info['total_multas']} | Total: ${info['monto_total']:,}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_pos += line_height - 5
            
            # Mostrar primera multa como ejemplo
            if info["multas"]:
                primera_multa = info["multas"][0]
                cv2.putText(frame, f"Ultima: {primera_multa['tipo'][:25]}...", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
        else:
            # Patente sin multas (color verde)
            cv2.putText(frame, "SIN MULTAS", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            y_pos += line_height
            cv2.putText(frame, "Patente limpia", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def procesar_frame(self, frame):
        """Procesa un frame del video según el estado actual"""
        
        # Estado: COMPLETADO - Mostrar resultado y luego reiniciar automáticamente
        if self.estado == "COMPLETADO":
            tiempo_transcurrido = time.time() - self.tiempo_resultado
            tiempo_restante = self.duracion_mostrar - tiempo_transcurrido
            
            if tiempo_restante > 0:
                # Dibujar información de multas
                frame = self.dibujar_info_multas(frame)
                
                # Contador regresivo
                cv2.putText(frame, f"Nueva busqueda en: {tiempo_restante:.1f}s", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
                self.consultar_multas(resultado)  # Consultar multas aquí
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
    
    print("📷 Detector de patentes con consulta de multas iniciado")
    print("🎯 Posiciona la patente frente a la cámara")
    print("🔄 El detector buscará automáticamente nuevas patentes")
    print("📋 Base de datos de multas cargada")
    print("⌨️ Presiona 'q' para salir")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No se pudo capturar el frame")
            break
        
        # Procesar frame
        frame_procesado, terminado = detector.procesar_frame(frame)
        
        # Mostrar frame
        cv2.imshow("Detector de Patentes - Con Consulta de Multas", frame_procesado)
        
        # Control de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()