#!/usr/bin/env python3
import argparse
import io
import math
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import segno
from reportlab.lib.pagesizes import portrait
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.graphics.barcode import code128


# ========= LAYOUT CONSTANTS (mirroring your browser app) =========

LABEL_W_MM = 36
LABEL_H_MM = 76
DPI = 300  # for parity with your canvas, mostly for intuition

# Fonts (from Index.html: HEAD_PT = 8, BODY_PT = 7, BIG_NUM_PT = 16)
HEAD_PT = 8.0
BODY_PT = 7.0
BIG_NUM_PT = 16.0

# Margins & sizes (taken from your JS and approximated where needed)
PAD_MM = 3.0                 # general padding
DM_SIZE_MM = 16.0            # square DM box (including quiet zone)
DM_QUIET_MM = 1.0            # quiet zone margin
DM_LEFT_PAD_MM = 1.0         # small offset from left for DM
TOP_TEXT_OFFSET_MM = 2.0     # offset from top of DM to SKU small text
UID_OFFSET_BELOW_TOP_MM = 2  # distance from bottom of top block to UID
BARCODE_TEXT_GAP_MM = 0.8    # gap between barcode and human-readable digits
DIVIDER_GAP_MM = 1.0         # gap between HR text and divider
BOTTOM_TEXT_TOP_GAP_MM = 1.0 # gap between divider and product text
BOTTOM_DM_BOTTOM_PAD_MM = 3.0  # space from bottom edge to bottom DM

# Derived page size in points for ReportLab
PAGE_W = LABEL_W_MM * mm
PAGE_H = LABEL_H_MM * mm


# ========= HELPERS =========

def split_sku(sku: str):
    """Split SKU into (small, big) per JS: last 3 digits as big."""
    sku = (sku or "").strip()
    if len(sku) > 3:
        return sku[:-3], sku[-3:]
    return "", sku  # if short, treat as big only


def make_dm_image(payload: str, box_size_pt: float) -> ImageReader:
    """Generate a DataMatrix PNG in memory and wrap as ImageReader."""
    if not payload:
        return None
    qr = segno.make(payload, micro=False, encoding="utf-8")
    buf = io.BytesIO()
    # border=0, we manage quiet zone ourselves
    qr.save(buf, kind="png", border=0, scale=1)
    buf.seek(0)
    return ImageReader(buf)


def draw_datamatrix(c: canvas.Canvas, img: ImageReader,
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
                 bar_height_mm: float = 20.0):
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


# ========= SINGLE LABEL DRAW =========

def draw_single_label(c: canvas.Canvas, row: pd.Series):
    """
    Draw a single label page on an existing canvas.
    Coordinates: ReportLab origin bottom-left.
    We'll mirror the JS layout as closely as possible.
    """

    # Extract fields
    po = str(row.get("PO_Number", "")).strip()
    sku = str(row.get("SKU_Code", "")).strip()
    style = str(row.get("Style", "")).strip()
    ean = str(row.get("EAN_Code", "")).strip() or sku
    product = str(row.get("Product", "")).strip()
    color = str(row.get("Color", "")).strip()
    size = str(row.get("Size", "")).strip()
    uid = str(row.get("UID", "")).strip()

    # DataMatrix payload: STYLE-SKU;UID
    dm_payload = f"{style}-{sku};{uid}" if (style and sku and uid) else ""

    # Clear page (ReportLab doesn't have a "background"; page is white by default)
    c.setFillColorRGB(0, 0, 0)

    # Fonts
    c.setStrokeColorRGB(0, 0, 0)

    # Precompute DM image once (top & bottom)
    dm_img = make_dm_image(dm_payload, DM_SIZE_MM * mm) if dm_payload else None

    # ---- TOP SECTION ----
    pad_pt = PAD_MM * mm
    dm_size_pt = DM_SIZE_MM * mm

    # Top-left DM position (JS uses dmLeftPad and pad)
    top_dm_x = DM_LEFT_PAD_MM * mm
    top_dm_y = PAGE_H - pad_pt - dm_size_pt  # from top

    draw_datamatrix(c, dm_img, top_dm_x, top_dm_y, dm_size_pt)

    # Right block (SKU text) aligned to right edge
    sku_small, sku_big = split_sku(sku)
    right_pad_pt = pad_pt
    sku_col_x = PAGE_W - right_pad_pt

    # Baseline near top: JS uses pad + small offset
    # We'll approximate: top text baseline ~ top_dm_y + dm_size_pt - TOP_TEXT_OFFSET
    # Remember y=0 bottom; top_dm_y is already bottom of DM box (we set it as y); DM is drawn upward
    # So we want small text ~ near top of label minus some mm; approximate:
    right_block_y = PAGE_H - pad_pt - (BODY_PT * mm / 3.0)  # slightly below top

    c.setFont("Helvetica", BODY_PT)
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.5)

    # Draw small and big SKU
    c.setFont("Helvetica", BODY_PT)
    if sku_small:
        c.drawRightString(sku_col_x, right_block_y, sku_small)

    big_y = right_block_y - (BODY_PT * 0.0) - (BODY_PT * 1.4)  # 1.4 line spacing downward
    c.setFont("Helvetica", BIG_NUM_PT)
    if sku_big:
        c.drawRightString(sku_col_x, right_block_y - BODY_PT * 1.4, sku_big)
    elif sku:
        c.drawRightString(sku_col_x, right_block_y - BODY_PT * 1.4, sku)

    # UID centered below DM+numbers
    # We'll set it about 2mm below the bottom of the top DM block
    top_block_bottom = PAGE_H - pad_pt - dm_size_pt  # approximate bottom of DM block
    uid_y = top_block_bottom - UID_OFFSET_BELOW_TOP_MM * mm
    c.setFont("Helvetica", BODY_PT)
    c.drawCentredString(PAGE_W / 2.0, uid_y, uid)

    # ---- BARCODE SECTION ----
    # Under UID, centered; width almost full label
    bc_full_w = PAGE_W - pad_pt * 2
    bc_y = uid_y - BODY_PT * 1.6  # going downwards in ReportLab means subtract
    draw_barcode(c, ean or sku or "000", PAGE_W / 2.0, bc_y, bc_full_w)

    # Human-readable digits under barcode
    hr_y = bc_y - (BODY_PT * 1.2)
    human_text = ean or sku or ""
    if human_text:
        c.setFont("Helvetica", BODY_PT)
        c.drawCentredString(PAGE_W / 2.0, hr_y, human_text)

    # Divider line below human-readable text
    divider_y = hr_y - DIVIDER_GAP_MM * mm
    c.setLineWidth(0.3)
    c.line(pad_pt, divider_y, PAGE_W - pad_pt, divider_y)

    # ---- BOTTOM TEXT ----
    text_start_y = divider_y - BOTTOM_TEXT_TOP_GAP_MM * mm

    # Simple wrapping for Product into up to 2 lines
    c.setFont("Helvetica", BODY_PT)
    max_text_width = PAGE_W - 2 * pad_pt
    words = product.split()
    lines = []
    current = ""
    for w in words:
        trial = (current + " " + w).strip()
        if c.stringWidth(trial, "Helvetica", BODY_PT) <= max_text_width:
            current = trial
        else:
            lines.append(current)
            current = w
        if len(lines) == 2:
            break
    if current and len(lines) < 2:
        lines.append(current)

    y = text_start_y
    for line in lines:
        c.drawString(pad_pt, y, line)
        y -= BODY_PT * 1.2

    # Color
    if color:
        c.drawString(pad_pt, y, color)
        y -= BODY_PT * 1.2

    # Size
    if size:
        size_text = f"Size: {size}"
        c.drawString(pad_pt, y, size_text)
        y -= BODY_PT * 1.2

    # ---- BOTTOM DM + SKU ----
    dm_bottom_y = BOTTOM_DM_BOTTOM_PAD_MM * mm
    dm_bottom_x = DM_LEFT_PAD_MM * mm
    draw_datamatrix(c, dm_img, dm_bottom_x, dm_bottom_y, dm_size_pt)

    # Bottom-right SKU (mirroring top-right)
    bottom_right_block_y = dm_bottom_y + dm_size_pt - BODY_PT * 1.4
    c.setFont("Helvetica", BODY_PT)
    if sku_small:
        c.drawRightString(sku_col_x, bottom_right_block_y + BODY_PT * 1.4, sku_small)
    c.setFont("Helvetica", BIG_NUM_PT)
    if sku_big:
        c.drawRightString(sku_col_x, bottom_right_block_y, sku_big)
    elif sku:
        c.drawRightString(sku_col_x, bottom_right_block_y, sku)


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
