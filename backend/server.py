import io
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from render_labels import run_pdf_by_po  # our batch renderer


app = FastAPI()

# Allow Netlify + local dev (you can lock this down later if you want)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "UID label renderer running"}


@app.post("/render/pdf-by-po")
async def render_pdf_by_po(file: UploadFile = File(...)):
    """
    Receive uid_labels.xlsx, render PDFs by PO, and return a ZIP that contains:
      - One PDF per PO
      - The same uid_labels.xlsx inside the ZIP
    """
    filename = (file.filename or "").lower()
    if not (filename.endswith(".xls") or filename.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Please upload an Excel file (.xls or .xlsx).")

    # Work in a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Save uploaded Excel
        input_path = tmpdir_path / "uid_labels.xlsx"
        contents = await file.read()
        input_path.write_bytes(contents)

        # Output directory for PDFs
        out_dir = tmpdir_path / "pdfs"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Run the batch renderer (1 process is fine on Render free tier)
        try:
            run_pdf_by_po(input_path, out_dir, processes=1)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Render error: {e}")

        # Build ZIP in memory: PDFs + the Excel file
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as zipf:
            # Add PDFs by PO
            for pdf_path in sorted(out_dir.glob("*.pdf")):
                zipf.write(pdf_path, arcname=pdf_path.name)

            # Add the UID-level Excel (with a nice name)
            zipf.write(input_path, arcname="uid_labels.xlsx")

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="labels_by_po_with_uid_excel.zip"'
            },
        )


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
