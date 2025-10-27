import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocesar_imagen(ruta_imagen):
    """Preprocesa la imagen para mejorar OCR"""
    try:
        # Leer imagen
        img = cv2.imread(ruta_imagen)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        
        # Convertir a escala de grises
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtro para reducir ruido
        sin_ruido = cv2.medianBlur(gris, 3)
        
        # Mejorar contraste con CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contraste_mejorado = clahe.apply(sin_ruido)
        
        # Binarización
        _, binarizada = cv2.threshold(contraste_mejorado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binarizada
        
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {e}")
        return None

def extraer_texto_tesseract(imagen_procesada, idioma='spa'):
    """Extrae texto usando Tesseract"""
    try:
        # Verificar que tenemos imagen
        if imagen_procesada is None:
            return "Error: No hay imagen para procesar"
        
        # Convertir imagen de OpenCV a formato PIL
        imagen_pil = Image.fromarray(imagen_procesada)
        
        # Extraer texto con configuración optimizada
        texto = pytesseract.image_to_string(imagen_pil, lang=idioma, config='--oem 3 --psm 6')
        
        print(f"✅ Texto extraído ({len(texto)} caracteres)")
        return texto.strip()
        
    except Exception as e:
        print(f"❌ Error en OCR: {e}")
        return f"Error en OCR: {e}"


if __name__ == "__main__":

    ruta_imagen = "C:/Users/Desktop/Downloads/x.jpg"
    
    # Verificar que la imagen existe
    if not os.path.exists(ruta_imagen):
        print(f"❌ Error: La imagen no existe en {ruta_imagen}")
        exit()
    
    print(f"📁 Procesando: {ruta_imagen}")
    print(f"📏 Tamaño imagen: {os.path.getsize(ruta_imagen)} bytes")
    
    try:
        # 1. Preprocesar imagen
        print("🔄 Preprocesando imagen...")
        imagen_procesada = preprocesar_imagen(ruta_imagen)
        
        # 2. Extraer texto
        print("🔍 Extrayendo texto...")
        texto_extraido = extraer_texto_tesseract(imagen_procesada, 'spa')

        
        # Guardar resultado solo si hay texto
        if texto_extraido and len(texto_extraido) > 5:
            with open("resultado_ocr.txt", "w", encoding="utf-8") as f:
                f.write(texto_extraido)
            print("\n✅ Texto guardado en 'resultado_ocr.txt'")
        else:
            print("\n⚠️  No se guardó archivo - texto muy corto o vacío")
            
    except Exception as e:
        print(f"❌ Error general: {e}")