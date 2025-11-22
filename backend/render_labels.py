#!/usr/bin/env python3
import argparse
import io
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import segno
from reportlab.lib.pagesizes import portrait
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.graphics.barcode import code128


# ========= LAYOUT CONSTANTS (mirroring browser layout, tuned to golden label) =========

LABEL_W_MM = 36
LABEL_H_MM = 76

# Fonts
HEAD_PT = 8.0
BODY_PT = 7.0
BIG_NUM_PT = 16.0

# Margins & sizes
TOP_MARGIN_MM = 4.0           # top white margin
SIDE_MARGIN_MM = 3.0          # left/right white margin
DM_SIZE_MM = 14.0             # DataMatrix box size
DM_QUIET_MM = 1.0             # quiet zone inside DM box

UID_GAP_MM = 6.0              # gap from bottom of top DM to UID baseline
BARCODE_TOP_GAP_MM = 14.0     # gap from UID baseline to bottom of barcode
BARCODE_HEIGHT_MM = 18.0      # barcode bar height
HR_GAP_MM = 2.0               # gap from barcode bars to HR digits
DIVIDER_GAP_MM = 3.0          # gap from HR digits to divider line
TEXT_TOP_GAP_MM = 2.0         # gap from divider to first product line
BOTTOM_DM_BOTTOM_PAD_MM = 6.0 # bottom white margin under bottom DM

PAGE_W = LABEL_W_MM * mm
PAGE_H = LABEL_H_MM * mm


# ========= HELPERS =========

def split_sku(sku: str):
    """Split SKU into (small, big) per JS: last 3 digits as big."""
    sku = (sku or "").strip()
    if len(sku) > 3:
        return sku[:-3], sku[-3:]
    return "", sku  # if short, treat as big only


def make_dm_image(payload: str) -> ImageReader | None:
    """Generate a DataMatrix PNG in memory and wrap as ImageReader."""
    if not payload:
        return None
    qr = segno.make(payload, micro=False, encoding="utf-8")
    buf = io.BytesIO()
    # border=0, we manage quiet zone ourselves
    qr.save(buf, kind="png", border=0, scale=1)
    buf.seek(0)
    return ImageReader(buf)


def draw_datamatrix(c: canvas.Canvas, img: ImageReader | None,
                    x_pt: float, y_pt: float,
                    box_size_pt: float):
    """Draw DM image in a square box with quiet zone margin (like JS)."""
    if img is None:
        return
    inner = box_size_pt - DM_QUIET_MM * mm * 2
    iw, ih = img.getSize()
    # scale to fit inner box
    scale = min(inner / iw, inner / ih, 1.0)
    iw_scaled = iw * scale
    ih_scaled = ih * scale
    dx = x_pt + DM_QUIET_MM * mm + (inner - iw_scaled) / 2
    dy = y_pt + DM_QUIET_MM * mm + (inner - ih_scaled) / 2
    c.drawImage(img, dx, dy, width=iw_scaled, height=ih_scaled, mask='auto')


def draw_barcode(c: canvas.Canvas, payload: str,
                 x_center_pt: float, y_pt: float,
                 target_width_pt: float,
                 bar_height_mm: float = BARCODE_HEIGHT_MM):
    """Draw Code128 barcode centered at x_center, with target width."""
    if not payload:
        payload = "000"

    bc = code128.Code128(payload, barHeight=bar_height_mm * mm, humanReadable=False)
    bc_width = bc.width

    if bc_width == 0:
        return

    scale_x = target_width_pt / bc_width

    c.saveState()
    c.translate(x_center_pt - (target_width_pt / 2.0), y_pt)
    c.scale(scale_x, 1.0)
    bc.drawOn(c, 0, 0)
    c.restoreState()


def wrap_text(c: canvas.Canvas, text: str, max_width_pt: float,
              font_name: str = "Helvetica", font_size: float = BODY_PT,
              max_lines: int = 2):
    """Simple word-wrap into <= max_lines."""
    words = (text or "").split()
    if not words:
        return []

    c.setFont(font_name, font_size)
    lines = []
    current = ""

    for w in words:
        trial = (current + " " + w).strip()
        if c.stringWidth(trial, font_name, font_size) <= max_width_pt:
            current = trial
        else:
            if current:
                lines.append(current)
            current = w
            if len(lines) >= max_lines:
                break

    if current and len(lines) < max_lines:
        lines.append(current)

    return lines


# ========= SINGLE LABEL DRAW =========

def draw_single_label(c: canvas.Canvas, row: pd.Series):
    """
    Draw a single label page on an existing canvas.
    Coordinates: ReportLab origin bottom-left.
    """

    # Extract fields
    sku = str(row.get("SKU_Code", "")).strip()
    style = str(row.get("Style", "")).strip()
    ean = str(row.get("EAN_Code", "")).strip() or sku
    product = str(row.get("Product", "")).strip()
    color = str(row.get("Color", "")).strip()
    size = str(row.get("Size", "")).strip()
    uid = str(row.get("UID", "")).strip()

    # DataMatrix payload: STYLE-SKU;UID
    dm_payload = f"{style}-{sku};{uid}" if (style and sku and uid) else ""

    c.setFillColorRGB(0, 0, 0)
    c.setStrokeColorRGB(0, 0, 0)

    dm_img = make_dm_image(dm_payload)

    # ---- TOP DM ----
    dm_size_pt = DM_SIZE_MM * mm
    top_dm_x = SIDE_MARGIN_MM * mm
    top_dm_y = PAGE_H - TOP_MARGIN_MM * mm - dm_size_pt
    draw_datamatrix(c, dm_img, top_dm_x, top_dm_y, dm_size_pt)

    # ---- TOP SKU (small above big, right-aligned) ----
    sku_small, sku_big = split_sku(sku)
    sku_x = PAGE_W - SIDE_MARGIN_MM * mm

    c.setFont("Helvetica", BODY_PT)
    small_y = PAGE_H - TOP_MARGIN_MM * mm
    if sku_small:
        c.drawRightString(sku_x, small_y, sku_small)

    c.setFont("Helvetica", BIG_NUM_PT)
    big_y = small_y - BIG_NUM_PT * 1.3
    if sku_big:
        c.drawRightString(sku_x, big_y, sku_big)
    elif sku:
        c.drawRightString(sku_x, big_y, sku)

    # ---- UID (centered, below DM block) ----
    uid_y = top_dm_y - UID_GAP_MM * mm
    c.setFont("Helvetica", BODY_PT)
    if uid:
        c.drawCentredString(PAGE_W / 2.0, uid_y, uid)

    # ---- BARCODE ----
    bc_full_w = PAGE_W - 2 * SIDE_MARGIN_MM * mm
    bc_y = uid_y - BARCODE_TOP_GAP_MM * mm
    draw_barcode(c, ean or sku or "000", PAGE_W / 2.0, bc_y, bc_full_w)

    # Human-readable digits
    hr_y = bc_y - HR_GAP_MM * mm
    human_text = ean or sku or ""
    if human_text:
        c.setFont("Helvetica", BODY_PT)
        c.drawCentredString(PAGE_W / 2.0, hr_y, human_text)

    # ---- DIVIDER ----
    divider_y = hr_y - DIVIDER_GAP_MM * mm
    c.setLineWidth(0.4)
    c.line(SIDE_MARGIN_MM * mm, divider_y,
           PAGE_W - SIDE_MARGIN_MM * mm, divider_y)

    # ---- PRODUCT / COLOR / SIZE ----
    text_y = divider_y - TEXT_TOP_GAP_MM * mm
    max_text_width = PAGE_W - 2 * SIDE_MARGIN_MM * mm

    c.setFont("Helvetica", BODY_PT)
    for line in wrap_text(c, product, max_text_width,
                          font_name="Helvetica", font_size=BODY_PT, max_lines=2):
        c.drawString(SIDE_MARGIN_MM * mm, text_y, line)
        text_y -= BODY_PT * 1.4

    if color:
        c.drawString(SIDE_MARGIN_MM * mm, text_y, color)
        text_y -= BODY_PT * 1.4

    if size:
        c.drawString(SIDE_MARGIN_MM * mm, text_y, f"Size: {size}")
        text_y -= BODY_PT * 1.4

    # ---- BOTTOM DM + SKU ----
    bottom_dm_y = BOTTOM_DM_BOTTOM_PAD_MM * mm
    bottom_dm_x = SIDE_MARGIN_MM * mm
    draw_datamatrix(c, dm_img, bottom_dm_x, bottom_dm_y, dm_size_pt)

    # bottom-right SKU (mirroring top-right)
    c.setFont("Helvetica", BODY_PT)
    bottom_small_y = bottom_dm_y + dm_size_pt - BODY_PT * 1.2
    if sku_small:
        c.drawRightString(sku_x, bottom_small_y, sku_small)

    c.setFont("Helvetica", BIG_NUM_PT)
    bottom_big_y = bottom_dm_y + dm_size_pt / 2.0
    if sku_big:
        c.drawRightString(sku_x, bottom_big_y, sku_big)
    elif sku:
        c.drawRightString(sku_x, bottom_big_y, sku)


# ========= BATCH PIPELINE =========

def load_uid_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str)
    required = ["PO_Number", "SKU_Code", "EAN_Code",
                "Product", "Color", "Size", "Style", "UID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Excel: {missing}")
    # Normalize to strings and strip
    for col in required:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def process_po_to_pdf(args_tuple):
    po_number, po_df, out_dir = args_tuple
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{po_number}.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=portrait((PAGE_W, PAGE_H)))

    for _, row in po_df.iterrows():
        draw_single_label(c, row)
        c.showPage()

    c.save()
    return str(pdf_path)


def run_pdf_by_po(input_xlsx: Path, out_dir: Path, processes: int = 0, po_filter=None):
    df = load_uid_excel(input_xlsx)
    if po_filter:
        allowed = set(po_filter)
        df = df[df["PO_Number"].isin(allowed)]

    groups = list(df.groupby("PO_Number", sort=False))
    jobs = [(po, g.copy(), out_dir) for po, g in groups]

    if not jobs:
        print("No labels to render.")
        return

    if processes and processes > 1:
        with mp.Pool(processes=processes) as pool:
            for result in pool.imap_unordered(process_po_to_pdf, jobs):
                print("Created:", result)
    else:
        for j in jobs:
            result = process_po_to_pdf(j)
            print("Created:", result)


# ========= CLI ENTRY =========

def main():
    parser = argparse.ArgumentParser(description="UID-level label batch renderer")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to uid_labels.xlsx")
    parser.add_argument("--out-dir", "-o", required=True,
                        help="Output directory (PDFs by PO)")
    parser.add_argument("--mode", choices=["pdf_by_po"], default="pdf_by_po",
                        help="Render mode (currently only pdf_by_po)")
    parser.add_argument("--processes", "-p", type=int, default=0,
                        help="Number of worker processes (0 => CPU count, 1 => no multiprocessing)")
    parser.add_argument("--po-filter", type=str, default="",
                        help="Optional comma-separated list of PO_Number to include (for testing)")

    args = parser.parse_args()
    input_xlsx = Path(args.input)
    out_dir = Path(args.out_dir)

    if args.mode == "pdf_by_po":
        procs = args.processes or mp.cpu_count()
        po_filter = [p.strip() for p in args.po_filter.split(",") if p.strip()] if args.po_filter else None
        run_pdf_by_po(input_xlsx, out_dir, processes=procs, po_filter=po_filter)
    else:
        raise SystemExit("Unsupported mode for now.")


if __name__ == "__main__":
    main()
