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

# Allowed front-end origins
origins = [
    "https://portrait3676uid.netlify.app",  # your Netlify site
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:4173",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or restrict to ["https://portrait3676uid.netlify.app"]
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/render/pdf-by-po")
async def render_pdf_by_po(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".xls", ".xlsx")):
        raise HTTPException(status_code=400, detail="Please upload an Excel file (.xls or .xlsx).")

    # Save uploaded Excel to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        in_path = tmpdir_path / file.filename
        with open(in_path, "wb") as f:
            f.write(await file.read())

        out_dir = tmpdir_path / "pdf_by_po"
        run_pdf_by_po(in_path, out_dir, processes=0)  # single process inside request

        # Zip all PDFs
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for pdf_path in sorted(out_dir.glob("*.pdf")):
                zf.write(pdf_path, arcname=pdf_path.name)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="po_pdfs.zip"'
            },
        )


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
