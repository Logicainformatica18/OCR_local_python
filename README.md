# 📝 OCR Server (GPU con EasyOCR) – Windows

Servidor OCR local hecho con **FastAPI** que usa **EasyOCR en GPU** para reconocer texto en **imágenes** y **PDFs**.  
Este README está documentado **paso a paso para Windows**.

---

## 🚀 Endpoints disponibles

- `GET /health`  
  Verifica si la GPU y Torch están disponibles. Devuelve nombre de GPU y versión de Torch.

- `POST /ocr/image`  
  Recibe una imagen (JPG/PNG/TIFF…) y devuelve el texto reconocido.

- `POST /ocr/pdf`  
  Convierte cada página de un PDF en imagen y devuelve el texto reconocido página por página.

---

## ⚙️ Requisitos en Windows

### Hardware
- GPU **NVIDIA** con soporte CUDA.  
- Drivers NVIDIA actualizados (mínimo versión compatible con tu GPU).  
- Al menos **2 GB de VRAM libre** (ideal 4 GB+).

### Software necesario
1. **Python 3.9 – 3.11**  
   Descárgalo de [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)  
   (marca la opción **"Add Python to PATH"** al instalar).
2. **Git**  
   Descárgalo de [https://git-scm.com/download/win](https://git-scm.com/download/win).
3. **Tesseract OCR**  
   Descarga desde [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) e instálalo.  
   Asegúrate de agregarlo al **PATH** (ej: `C:\Program Files\Tesseract-OCR`).
4. **Ghostscript**  
   Descarga desde [https://www.ghostscript.com/releases/gsdnld.html](https://www.ghostscript.com/releases/gsdnld.html).  
5. **qpdf** (solo si usarás `ocrmypdf`)  
   Descarga desde [https://sourceforge.net/projects/qpdf/](https://sourceforge.net/projects/qpdf/).

---

## 📦 Instalación paso a paso

1. **Clonar el repositorio**
   ```powershell
   git clone https://github.com/tuusuario/ocr_server.git
   cd ocr_server
   ```

2. **Crear entorno virtual**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Actualizar pip**
   ```powershell
   python -m pip install --upgrade pip
   ```

4. **Instalar dependencias principales**
   ```powershell
   pip install -r requirements.txt
   ```

5. **Instalar PyTorch con CUDA**  
   (ejemplo para CUDA 12.4):
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

👉 Si tu GPU usa otra versión de CUDA, consulta:  
https://pytorch.org/get-started/locally/

---

## ▶️ Ejecución del servidor

1. Activa el entorno virtual:
   ```powershell
   .\.venv\Scripts\activate
   ```

2. Ejecuta FastAPI con Uvicorn:
   ```powershell
   uvicorn ocr_server:app --host 0.0.0.0 --port 8000 --reload
   ```

3. Abre en tu navegador:
   - [http://localhost:8000/docs](http://localhost:8000/docs) → Swagger UI para probar los endpoints.  
   - [http://localhost:8000/health](http://localhost:8000/health) → estado de GPU y Torch.

---

## 🧪 Pruebas rápidas con curl

### 1. Verificar salud de la GPU
```powershell
curl http://localhost:8000/health
```

Ejemplo de respuesta:

```json
{
  "ok": true,
  "gpu": "NVIDIA GeForce RTX 3060",
  "torch": "2.4.1",
  "torch_cuda": "12.4",
  "python": "3.10.12"
}
```

### 2. Probar OCR en una imagen
```powershell
curl -X POST "http://localhost:8000/ocr/image" -F "file=@ejemplo.png" -F "lang=spa+eng"
```

Respuesta esperada:

```json
{
  "ok": true,
  "engine": "easyocr",
  "gpu": true,
  "pages": 1,
  "text": "Texto detectado en la imagen..."
}
```

### 3. Probar OCR en un PDF
```powershell
curl -X POST "http://localhost:8000/ocr/pdf" -F "file=@ejemplo.pdf" -F "lang=spa+eng"
```

Respuesta esperada:

```json
{
  "ok": true,
  "engine": "easyocr",
  "gpu": true,
  "pages": 3,
  "text": "Texto página 1...\n\nTexto página 2...\n\nTexto página 3..."
}
```