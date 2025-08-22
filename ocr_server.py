# ocr_server.py
# -------------------------------
# FastAPI local SOLO GPU (EasyOCR) para imágenes y PDFs
# - /ocr/image     : EasyOCR (GPU)
# - /ocr/pdf       : PDF -> imágenes + EasyOCR (GPU)
# -------------------------------

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile, os, sys
import fitz  # PyMuPDF
from PIL import Image
import numpy as np

# ---- GPU (DL) ----
import easyocr
import torch

app = FastAPI(title="Local OCR (GPU only EasyOCR)", version="2.0")

# --------- Utils ---------
_reader_cache = {}

def get_easyocr_reader(lang_list):
    """Crea/reusa un EasyOCR Reader, siempre GPU."""
    key = tuple(sorted(lang_list))
    if key in _reader_cache:
        return _reader_cache[key]

    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU no disponible en Torch. Revisa tu instalación de CUDA.")

    reader = easyocr.Reader(lang_list, gpu=True)
    _reader_cache[key] = reader
    return reader

def list_from_lang_str(lang_str: str):
    # "spa" o "spa+eng" -> ['es'] o ['es','en']
    mapping = {'spa': 'es', 'eng': 'en'}
    return [mapping.get(tok.strip().lower(), tok.strip().lower()) for tok in lang_str.split('+')]

# --------- Endpoints ---------
@app.get("/health")
def health():
    """Salud básica: estado de GPU."""
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
    return {
        "ok": cuda_ok,
        "gpu": gpu_name,
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "python": sys.version.split()[0],
    }

# ---------- Imagenes ----------
@app.post("/ocr/image")
async def ocr_image(
    file: UploadFile = File(...),
    lang: str = Form("spa")   # "spa" o "spa+eng"
):
    """OCR imágenes con EasyOCR (GPU obligatorio)."""
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(await file.read()); tmp.close()
    try:
        img = Image.open(tmp.name).convert("RGB")
        reader = get_easyocr_reader(list_from_lang_str(lang))
        result = reader.readtext(np.array(img), detail=1, paragraph=True)
        text = "\n".join([r[1] for r in result]) if result else ""
        return {"ok": True, "engine": "easyocr", "gpu": True, "pages": 1, "text": text}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        try: os.remove(tmp.name)
        except Exception: pass

# ---------- PDFs ----------
@app.post("/ocr/pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    lang: str = Form("spa"),
    max_pages: int = Form(0),
    dpi: int = Form(300)
):
    """
    Convierte PDF a imágenes y reconoce con EasyOCR (GPU).
    """
    pdf_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_tmp.write(await file.read()); pdf_tmp.close()
    img_paths = []
    try:
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

        reader = get_easyocr_reader(list_from_lang_str(lang))
        texts = []
        for path in img_paths:
            img = Image.open(path).convert("RGB")
            result = reader.readtext(np.array(img), detail=1, paragraph=True)
            texts.append("\n".join([r[1] for r in result]) if result else "")

        return {
            "ok": True,
            "engine": "easyocr",
            "gpu": True,
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
