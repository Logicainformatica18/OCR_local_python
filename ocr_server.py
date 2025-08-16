# ocr_server.py
# -------------------------------
# FastAPI local para OCR de imágenes y PDFs
# - /ocr/image     : Tesseract (CPU) para imágenes
# - /ocr/pdf       : Extrae texto nativo; si no hay, OCRmyPDF+Tesseract (CPU) y opcionalmente devuelve PDF buscable
# - /ocr/image_dl  : EasyOCR (GPU si disponible) para imágenes
# - /ocr/pdf_dl    : Render PDF->imagen (CPU) + EasyOCR (GPU si disponible), devuelve texto
#
# Requisitos (Windows):
#   - Tesseract (con Spanish spa)  -> tesseract --version
#   - Ghostscript                  -> gswin64c -v
# Paquetes Python:
#   fastapi, uvicorn[standard], python-multipart, pytesseract, Pillow, pymupdf, ocrmypdf
#   (para GPU) torch+CUDA (cu124 recomendado), easyocr, numpy
# -------------------------------

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile, os, sys, subprocess, base64
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from typing import Tuple, Dict, Any

# ---- GPU (DL) imports opcionales ----
import numpy as np
try:
    import easyocr
    HAS_EASYOCR = True
except Exception:
    HAS_EASYOCR = False

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# (Opcional) fija la ruta a Tesseract si NO está en PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

app = FastAPI(title="Local OCR (Tesseract + OCRmyPDF + EasyOCR GPU)", version="1.2")

# --------- Utilidades comunes ---------
def extract_text_native_pdf(pdf_path: str) -> Tuple[str, int]:
    """Extrae texto 'copiable' de un PDF (sin OCR). Devuelve (texto, páginas)."""
    doc = fitz.open(pdf_path)
    texts = []
    for p in doc:
        t = p.get_text("text").strip()
        if t:
            texts.append(t)
    pages = len(doc)
    doc.close()
    return "\n\n".join(texts), pages

def tesseract_ocr_pil(img: Image.Image, lang: str = "spa", config: str = "--oem 3 --psm 6") -> str:
    """OCR sobre una imagen PIL (escala de grises) con Tesseract (CPU)."""
    gray = img.convert("L")
    return pytesseract.image_to_string(gray, lang=lang, config=config)

def bin_path_exists(exe: str) -> bool:
    """Comprueba si un ejecutable del sistema está disponible en PATH."""
    from shutil import which
    return which(exe) is not None

# --------- Utilidades Torch/EasyOCR ---------
def torch_env_info() -> Dict[str, Any]:
    """Datos de Torch/CUDA para reporte."""
    info: Dict[str, Any] = {
        "torch": HAS_TORCH,
        "torch_version": None,
        "cuda": False,
        "cuda_version": None,
        "gpu_name": None,
        "device_count": 0,
    }
    if not HAS_TORCH:
        return info
    try:
        info["torch_version"] = getattr(torch, "__version__", None)
        info["cuda"] = torch.cuda.is_available()
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if info["cuda"]:
            info["device_count"] = torch.cuda.device_count()
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    except Exception:
        pass
    return info

_reader_cache = {}

def get_easyocr_reader(lang_list, gpu_preference=None):
    """
    Crea o reutiliza un EasyOCR Reader.
    lang_list: lista de idiomas, ej ['es'] o ['es','en']
    gpu_preference: None => intenta GPU y cae a CPU; True => fuerza GPU; False => CPU
    """
    if not HAS_EASYOCR:
        raise RuntimeError("EasyOCR no está instalado. Ejecuta: pip install easyocr numpy")

    key = (tuple(sorted(lang_list)), gpu_preference)
    if key in _reader_cache:
        return _reader_cache[key]

    # auto: intenta GPU y si falla, usa CPU
    if gpu_preference is None:
        try:
            reader = easyocr.Reader(lang_list, gpu=True)
        except Exception:
            reader = easyocr.Reader(lang_list, gpu=False)
    else:
        reader = easyocr.Reader(lang_list, gpu=bool(gpu_preference))

    _reader_cache[key] = reader
    return reader

def list_from_lang_str(lang_str: str):
    # "spa" o "spa+eng" -> ['es'] o ['es','en'] (EasyOCR usa 'es'/'en')
    mapping = {'spa': 'es', 'eng': 'en'}
    return [mapping.get(tok.strip().lower(), tok.strip().lower()) for tok in lang_str.split('+')]


def reader_uses_cuda(reader) -> Dict[str, Any]:
    """Devuelve {'gpu': bool, 'device': 'cuda:0'|'cpu'} según el reader de EasyOCR."""
    device_str = None
    is_cuda = False
    try:
        dev = getattr(reader, 'device', None)  # torch.device o str
        if dev is not None:
            device_str = str(dev)
            # dev puede ser torch.device('cuda:0') o 'cpu'
            if isinstance(dev, str):
                is_cuda = dev.startswith('cuda')
            else:
                is_cuda = getattr(dev, 'type', '') == 'cuda'
    except Exception:
        pass
    return {"gpu": is_cuda, "device": device_str}

# --------- Endpoints ---------
@app.get("/health")
def health():
    """Salud básica: verifica binarios clave y estado de DL."""
    tesseract_ok = bin_path_exists("tesseract")
    gs_ok = bin_path_exists("gswin64c") or bin_path_exists("gs")  # Windows/Linux
    easyocr_ok = HAS_EASYOCR
    tinfo = torch_env_info()
    return {
        "ok": True,
        "tesseract": tesseract_ok,
        "ghostscript": gs_ok,
        "easyocr": easyocr_ok,
        **tinfo,
    }

# ---------- CPU: Tesseract imagen ----------
@app.post("/ocr/image")
async def ocr_image(
    file: UploadFile = File(...),
    lang: str = Form("spa"),
    psm: int = Form(6),        # 6=parrafos; 4=columnas; 7=linea; 3=auto
    oem: int = Form(3)         # 3=default, 1=LSTM only
):
    """OCR para imágenes con Tesseract (CPU)."""
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(await file.read())
    tmp.close()
    try:
        img = Image.open(tmp.name).convert("RGB")
        cfg = f"--oem {oem} --psm {psm}"
        txt = tesseract_ocr_pil(img, lang=lang, config=cfg)
        return {"ok": True, "engine": "tesseract", "pages": 1, "text": txt}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        try: os.remove(tmp.name)
        except Exception: pass

# ---------- CPU: OCRmyPDF (PDF buscable) ----------
@app.post("/ocr/pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    lang: str = Form("spa"),
    force_ocr: bool = Form(False),           # True => ignora texto nativo
    jobs: int = Form(1),                     # hilos para ocrmypdf (modo suave)
    return_searchable_pdf: bool = Form(False),
    return_pdf_as_base64: bool = Form(False) # si true y return_searchable_pdf, devuelve base64 (ojo tamaño)
):
    """
    OCR para PDFs (CPU):
      1) Si no force_ocr: intenta extraer texto nativo (rápido).
      2) Si no hay texto: OCRmyPDF (Tesseract + preprocesado) y extraer texto del PDF resultante.
    """
    pdf_in = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_in.write(await file.read())
    pdf_in.close()

    out_pdf = None
    try:
        if not force_ocr:
            native_text, pages = extract_text_native_pdf(pdf_in.name)
            if native_text and len(native_text.split()) > 10:
                return {"ok": True, "engine": "native", "pages": pages, "text": native_text}

        # OCR con OCRmyPDF (CPU). Requiere Tesseract y Ghostscript instalados
        out_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        cmd = [
            sys.executable, "-m", "ocrmypdf",
            "-l", lang,
            "--jobs", str(max(1, jobs)),
            "--force-ocr",
            "--optimize", "0",
            pdf_in.name, out_pdf
        ]

        # Limitar hilos internos de Tesseract (OpenMP) para bajar consumo en laptops
        env = os.environ.copy()
        env["OMP_THREAD_LIMIT"] = env.get("OMP_THREAD_LIMIT", "1")

        run = subprocess.run(cmd, capture_output=True, env=env)
        if run.returncode != 0:
            err = (run.stderr or run.stdout).decode("utf-8", "ignore")
            raise RuntimeError(f"OCRmyPDF error: {err}")

        text_ocr, pages2 = extract_text_native_pdf(out_pdf)

        resp = {
            "ok": True,
            "engine": "tesseract+ocrmypdf",
            "pages": pages2,
            "text": text_ocr
        }

        if return_searchable_pdf:
            if return_pdf_as_base64:
                with open(out_pdf, "rb") as f:
                    resp["pdf_b64"] = base64.b64encode(f.read()).decode("ascii")
                try: os.remove(out_pdf)
                except Exception: pass
            else:
                resp["pdf_path"] = out_pdf  # ruta local temporal (pruebas)
                out_pdf = None  # conserva archivo

        return resp

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        try: os.remove(pdf_in.name)
        except Exception: pass
        if out_pdf:
            try: os.remove(out_pdf)
            except Exception: pass

# ---------- GPU/CPU: EasyOCR imagen (deep learning) ----------
@app.post("/ocr/image_dl")
async def ocr_image_dl(
    file: UploadFile = File(...),
    lang: str = Form("spa"),            # acepta "spa" o "spa+eng"
    use_gpu: bool | None = Form(None)   # True/False/None(auto)
):
    """OCR para imágenes con EasyOCR (GPU si disponible). Devuelve texto."""
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(await file.read()); tmp.close()
    try:
        img = Image.open(tmp.name).convert("RGB")
        reader = get_easyocr_reader(list_from_lang_str(lang), gpu_preference=use_gpu)
        result = reader.readtext(np.array(img), detail=1, paragraph=True)
        text = "\n".join([r[1] for r in result]) if result else ""
        devinfo = reader_uses_cuda(reader)
        tinfo = torch_env_info()
        return {
            "ok": True,
            "engine": "easyocr",
            "gpu": devinfo["gpu"],
            "device": devinfo["device"],
            "torch_cuda": tinfo.get("cuda", False),
            "pages": 1,
            "text": text,
        }
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        try: os.remove(tmp.name)
        except Exception: pass

# ---------- GPU/CPU: EasyOCR PDF (deep learning) ----------
@app.post("/ocr/pdf_dl")
async def ocr_pdf_dl(
    file: UploadFile = File(...),
    lang: str = Form("spa"),
    use_gpu: bool | None = Form(None),
    max_pages: int = Form(0),   # 0 = todas; usa 3 para pruebas
    dpi: int = Form(300)        # 300 suele ir bien
):
    """
    Convierte PDF a imágenes (CPU) y reconoce con EasyOCR (GPU si disponible). Devuelve texto.
    No genera PDF 'buscable'; para eso usa /ocr/pdf.
    """
    pdf_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_tmp.write(await file.read()); pdf_tmp.close()
    img_paths = []
    try:
        # Render PDF->imagen (CPU) con PyMuPDF
        doc = fitz.open(pdf_tmp.name)
        limit = doc.page_count if max_pages <= 0 else min(max_pages, doc.page_count)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for i in range(limit):
            p = doc.load_page(i)
            pix = p.get_pixmap(matrix=mat, alpha=False)
            path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            pix.save(path)
            img_paths.append(path)
        doc.close()

        reader = get_easyocr_reader(list_from_lang_str(lang), gpu_preference=use_gpu)
        texts = []
        for path in img_paths:
            img = Image.open(path).convert("RGB")
            result = reader.readtext(np.array(img), detail=1, paragraph=True)
            texts.append("\n".join([r[1] for r in result]) if result else "")

        devinfo = reader_uses_cuda(reader)
        tinfo = torch_env_info()
        return {
            "ok": True,
            "engine": "easyocr",
            "gpu": devinfo["gpu"],
            "device": devinfo["device"],
            "torch_cuda": tinfo.get("cuda", False),
            "pages": len(texts),
            "text": "\n\n".join(texts)
        }
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        try: os.remove(pdf_tmp.name)
        except Exception: pass
        for pth in img_paths:
            try: os.remove(pth)
            except Exception: pass
