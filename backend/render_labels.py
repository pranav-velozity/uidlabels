#!/usr/bin/env python3
import argparse
import io
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from pystrich.datamatrix import DataMatrixEncoder
from reportlab.lib.pagesizes import portrait
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.graphics.barcode import code128

# ========= LAYOUT CONSTANTS =========

LABEL_W_MM = 36
LABEL_H_MM = 76

# Fonts (all bold to match browser)
HEAD_FONT = "Helvetica-Bold"
BODY_FONT = "Helvetica-Bold"
BIG_FONT = "Helvetica-Bold"

HEAD_PT = 8.0
BODY_PT = 7.0
BIG_NUM_PT = 16.0

# Margins & sizes
TOP_MARGIN_MM = 4.0            # (still used for other spacing, not top DM)
SIDE_MARGIN_MM = 3.0           # left/right white margin for text
BC_SIDE_MARGIN_MM = 0.2        # no side margin => max barcode width
DM_SIZE_MM = 20.0              # DataMatrix box size (was 14.0; +50% taller/wider)
DM_QUIET_MM = 1.0              # quiet zone inside DM box

UID_GAP_MM = 4.5               # gap from bottom of top DM to UID baseline
BARCODE_TOP_GAP_MM = 7.0      # gap from UID to barcode (moves barcode up)
BARCODE_HEIGHT_MM = 5.0        # shorter bars => visually thinner barcode
HR_GAP_MM = 2.5                # gap from bars to 13-digit text
DIVIDER_GAP_MM = 2.5           # gap from HR digits to divider line
TEXT_TOP_GAP_MM = 5.0          # gap from divider to first product line
BOTTOM_DM_BOTTOM_PAD_MM = 1.0  # bottom white margin under bottom DM

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
    """
    Generate a DataMatrix PNG in memory and wrap as ImageReader.

    Uses pyStrich's DataMatrixEncoder so this is a true DataMatrix symbol,
    not a QR code.
    """
    payload = (payload or "").strip()
    if not payload:
        return None

    # DataMatrixEncoder returns PNG bytes via get_imagedata().
    # cellsize controls how many pixels per module; 2 is usually a good
    # balance between resolution and size. We scale again in draw_datamatrix.
    try:
        encoder = DataMatrixEncoder(payload)
        png_bytes = encoder.get_imagedata(cellsize=2)
    except Exception:
        # If anything goes wrong, fail silently for this label
        return None

    buf = io.BytesIO(png_bytes)
    buf.seek(0)
    return ImageReader(buf)


def draw_datamatrix(c: canvas.Canvas, img: ImageReader | None,
                    x_pt: float, y_pt: float,
                    box_size_pt: float):
    """Draw DM image filling the box (no extra margins here)."""
    if img is None:
        return

    # Stretch the image to exactly the square box. The PNG already
    # includes its own quiet zone, so we don't add another one.
    c.drawImage(
        img,
        x_pt,
        y_pt,
        width=box_size_pt,
        height=box_size_pt,
        preserveAspectRatio=True,
        mask="auto",
    )

def draw_barcode(c: canvas.Canvas, payload: str,
                 x_center_pt: float, y_pt: float,
                 target_width_pt: float,
                 bar_height_mm: float = BARCODE_HEIGHT_MM):
    """Draw Code128 barcode centered at x_center, scaled to target width."""
    if not payload:
        payload = "000"

    bc = code128.Code128(payload, barHeight=bar_height_mm * mm, humanReadable=False)
    bc_width = bc.width
    if bc_width == 0:
        return

    scale_x = target_width_pt / bc_width

    c.saveState()
    # Use dark gray instead of pure black for a lighter visual weight
    c.setFillColorRGB(0.45, 0.45, 0.45)
    c.setStrokeColorRGB(0.45, 0.45, 0.45)
    c.translate(x_center_pt - (target_width_pt / 2.0), y_pt)
    c.scale(scale_x, 1.0)
    bc.drawOn(c, 0, 0)
    c.restoreState()


def wrap_text(c: canvas.Canvas, text: str, max_width_pt: float,
              font_name: str = BODY_FONT, font_size: float = BODY_PT,
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
    top_dm_x = 0 * mm  # flush to left edge of label
    top_dm_y = PAGE_H - dm_size_pt  # flush to top edge (no top white margin)
    draw_datamatrix(c, dm_img, top_dm_x, top_dm_y, dm_size_pt)

    # ---- TOP SKU (small above big, more spaced, centred on DM) ----
    sku_small, sku_big = split_sku(sku)
    sku_x = PAGE_W - SIDE_MARGIN_MM * mm

    center_y_dm_top = top_dm_y + dm_size_pt / 2.0

    # Big number baseline a bit below DM centre
    c.setFont(BIG_FONT, BIG_NUM_PT)
    big_y_top = center_y_dm_top - BIG_NUM_PT * 0.4
    if sku_big:
        c.drawRightString(sku_x, big_y_top, sku_big)
    elif sku:
        c.drawRightString(sku_x, big_y_top, sku)

    # Small number baseline above big (more gap now)
    c.setFont(BODY_FONT, BODY_PT)
    small_y_top = big_y_top + 5.0 * mm  # was 4mm; more separation
    if sku_small:
        c.drawRightString(sku_x, small_y_top, sku_small)

    # ---- UID (centered, below DM block) ----
    uid_y = top_dm_y - UID_GAP_MM * mm
    c.setFont(BODY_FONT, BODY_PT)
    if uid:
        c.drawCentredString(PAGE_W / 2.0, uid_y, uid)

    # ---- BARCODE (shorter, wider) ----
    bc_full_w = 1.3 * (PAGE_W - 2 * BC_SIDE_MARGIN_MM * mm)
    bc_y = uid_y - BARCODE_TOP_GAP_MM * mm
    draw_barcode(c, ean or sku or "000", PAGE_W / 2.0, bc_y, bc_full_w)

    # Human-readable digits (more gap below bars)
    hr_y = bc_y - HR_GAP_MM * mm
    human_text = ean or sku or ""
    if human_text:
        c.setFont(BODY_FONT, BODY_PT)
        c.drawCentredString(PAGE_W / 2.0, hr_y, human_text)

    # ---- DIVIDER ----
    divider_y = hr_y - DIVIDER_GAP_MM * mm
    c.setLineWidth(0.4)
    c.line(
        0,
        divider_y,
        PAGE_W,
        divider_y,
    )

    # ---- PRODUCT / COLOR / SIZE ----
    text_y = divider_y - TEXT_TOP_GAP_MM * mm
    max_text_width = PAGE_W - 2 * SIDE_MARGIN_MM * mm

    c.setFont(BODY_FONT, BODY_PT)
    for line in wrap_text(c, product, max_text_width,
                          font_name=BODY_FONT, font_size=BODY_PT, max_lines=2):
        c.drawString(SIDE_MARGIN_MM * mm, text_y, line)
        text_y -= BODY_PT * 1.4

    if color:
        c.drawString(SIDE_MARGIN_MM * mm, text_y, color)
        text_y -= BODY_PT * 1.4

    if size:
        c.drawString(SIDE_MARGIN_MM * mm, text_y, f"Size: {size}")
        text_y -= BODY_PT * 1.4

    # ---- BOTTOM DM ----
    bottom_dm_y = BOTTOM_DM_BOTTOM_PAD_MM * mm
    bottom_dm_x = 0 * mm
    draw_datamatrix(c, dm_img, bottom_dm_x, bottom_dm_y, dm_size_pt)

    # ---- BOTTOM SKU (mirror top spacing/centering) ----
    center_y_dm_bottom = bottom_dm_y + dm_size_pt / 2.0
    sku_x_bottom = sku_x

    c.setFont(BIG_FONT, BIG_NUM_PT)
    big_y_bottom = center_y_dm_bottom - BIG_NUM_PT * 0.4
    if sku_big:
        c.drawRightString(sku_x_bottom, big_y_bottom, sku_big)
    elif sku:
        c.drawRightString(sku_x_bottom, big_y_bottom, sku)

    c.setFont(BODY_FONT, BODY_PT)
    small_y_bottom = big_y_bottom + 5.0 * mm  # match top gap
    if sku_small:
        c.drawRightString(sku_x_bottom, small_y_bottom, sku_small)


# ========= BATCH PIPELINE =========

def load_uid_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str)
    required = ["PO_Number", "SKU_Code", "EAN_Code",
                "Product", "Color", "Size", "Style", "UID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Excel: {missing}")
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
