#!/usr/bin/env python3
import io
import tempfile
import zipfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from render_labels import run_pdf_by_po  # import from same repo


app = FastAPI()

# --- CORS: wide open for now (we can tighten later) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow everything while we debug
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "uid-label-batch-renderer"}


@app.post("/render/pdf-by-po")
async def render_pdf_by_po_endpoint(file: UploadFile = File(...)):
    # Basic validation
    if not file.filename.lower().endswith((".xls", ".xlsx")):
        raise HTTPException(status_code=400, detail="Please upload an Excel file (.xls or .xlsx).")

    try:
        # Save uploaded Excel to a temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            in_path = tmpdir_path / file.filename

            with open(in_path, "wb") as f:
                f.write(await file.read())

            out_dir = tmpdir_path / "pdf_by_po"
            # single process inside request
            run_pdf_by_po(in_path, out_dir, processes=0)

            # Zip all PDFs
            pdf_files = list(out_dir.glob("*.pdf"))
            if not pdf_files:
                raise HTTPException(status_code=500, detail="No PDFs were generated.")

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for pdf_path in pdf_files:
                    zf.write(pdf_path, arcname=pdf_path.name)
            buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="po_pdfs.zip"'
            },
        )

    except HTTPException:
        # Let explicit HTTP errors pass through
        raise
    except Exception as e:
        # Log to Render logs and return a generic 500
        print("ERROR in /render/pdf-by-po:", repr(e))
        raise HTTPException(status_code=500, detail="Internal server error during PDF generation.")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
