#!/usr/bin/env python3
"""
Clean, fully-space-indented label renderer.

- 36 x 76 mm portrait label.
- Uses DataMatrix (via segno.helpers.make_data_matrix).
- DataMatrix size: 25 mm (top-left and bottom-left).
- Barcode is Code128, stretched wide, with dark gray bars.
- Layout matches the browser preview used by index.html:
  * Front-end expands rows by Labels_Count and generates a UID per label.
  * Backend Excel contains columns:
      PO_Number, SKU_Code, EAN_Code, Product, Color, Size, Style, UID, (optional) Label_Index
  * This module only cares about the 8 required columns and ignores extras.
"""

import argparse
import io
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from segno import helpers
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.graphics.barcode import code128

# ========= LAYOUT CONSTANTS =========

# Label physical size (in millimetres)
LABEL_W_MM: float = 36.0
LABEL_H_MM: float = 76.0

# Fonts (all bold to match browser)
HEAD_FONT: str = "Helvetica-Bold"
BODY_FONT: str = "Helvetica-Bold"
BIG_FONT: str = "Helvetica-Bold"

HEAD_PT: float = 8.0
BODY_PT: float = 7.0
BIG_NUM_PT: float = 16.0

# Margins & sizes
TOP_MARGIN_MM: float = 4.0             # top margin
SIDE_MARGIN_MM: float = 3.0            # left/right text margin
DM_LEFT_MM: float = 1.0                # left offset for DataMatrix (near edge)
DM_SIZE_MM: float = 25.0               # DataMatrix box size (top + bottom)
DM_QUIET_MM: float = 1.0               # quiet zone inside DM box

# Vertical distances
UID_GAP_MM: float = 5.0                # gap from bottom of top DM to UID
BARCODE_TOP_GAP_MM: float = 8.0        # gap from UID to barcode
BARCODE_HEIGHT_MM: float = 7.0         # bar height
HR_GAP_MM: float = 3.0                 # gap from bars to human-readable digits
DIVIDER_GAP_MM: float = 3.0            # gap from HR digits to divider
TEXT_TOP_GAP_MM: float = 3.0           # gap from divider to first product line
BOTTOM_DM_BOTTOM_PAD_MM: float = 2.0   # space from bottom edge to bottom DM

# Barcode horizontal margin (small so it stretches wide)
BC_SIDE_MARGIN_MM: float = 0.5

PAGE_W: float = LABEL_W_MM * mm
PAGE_H: float = LABEL_H_MM * mm


# ========= HELPERS =========

def split_sku(sku: str) -> Tuple[str, str]:
    """
    Split SKU into (small, big) per browser behaviour:
    - last 3 characters are the "big" suffix
    - everything before that is the "small" prefix
    """
    sku = (sku or "").strip()
    if len(sku) > 3:
        return sku[:-3], sku[-3:]
    return "", sku  # if short, treat as big only


def make_dm_image(payload: str):
    """Generate a DataMatrix PNG in memory and wrap as ImageReader."""
    if not payload:
        return None
    dm = helpers.make_data_matrix(payload)  # DataMatrix, not QR
    buf = io.BytesIO()
    # border=0: we manage quiet zone ourselves; scale=1: we handle sizing in draw_datamatrix
    dm.save(buf, kind="png", border=0, scale=1)
    buf.seek(0)
    return ImageReader(buf)


def draw_datamatrix(
    c: canvas.Canvas,
    img,
    x_pt: float,
    y_pt: float,
    box_size_pt: float,
) -> None:
    """
    Draw DM image inside a square box with a quiet zone margin.
    (x_pt, y_pt) is the bottom-left of the outer DM box.
    """
    if img is None:
        return

    inner = box_size_pt - DM_QUIET_MM * mm * 2.0
    iw, ih = img.getSize()
    if iw <= 0 or ih <= 0 or inner <= 0:
        return

    scale = min(inner / iw, inner / ih)
    iw_scaled = iw * scale
    ih_scaled = ih * scale

    dx = x_pt + DM_QUIET_MM * mm + (inner - iw_scaled) / 2.0
    dy = y_pt + DM_QUIET_MM * mm + (inner - ih_scaled) / 2.0

    c.drawImage(img, dx, dy, width=iw_scaled, height=ih_scaled, mask="auto")


def draw_barcode(
    c: canvas.Canvas,
    payload: str,
    x_center_pt: float,
    y_pt: float,
    target_width_pt: float,
    bar_height_mm: float = BARCODE_HEIGHT_MM,
) -> None:
    """
    Draw Code128 barcode centered at x_center_pt, scaled to target width.

    Bars are drawn in a dark gray (not pure black) to match the design.
    """
    if not payload:
        payload = "000"

    bc = code128.Code128(payload, barHeight=bar_height_mm * mm, humanReadable=False)
    bc_width = bc.width
    if bc_width <= 0:
        return

    # Scale to fill desired width
    scale_x = target_width_pt / bc_width

    c.saveState()

    # Lighter than pure black, but still very scannable
    c.setFillColorRGB(0.12, 0.12, 0.12)
    c.setStrokeColorRGB(0.12, 0.12, 0.12)

    # Position and scale
    c.translate(x_center_pt - (target_width_pt / 2.0), y_pt)
    c.scale(scale_x, 1.0)
    bc.drawOn(c, 0, 0)

    c.restoreState()


def wrap_text(
    c: canvas.Canvas,
    text: str,
    max_width_pt: float,
    font_name: str = BODY_FONT,
    font_size: float = BODY_PT,
    max_lines: int = 2,
) -> List[str]:
    """Simple word-wrap into <= max_lines using the canvas for width measurement."""
    words = (text or "").split()
    if not words:
        return []

    c.setFont(font_name, font_size)
    lines: List[str] = []
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
            else:
                # Single very long word: force-place it
                lines.append(w)
                current = ""
                if len(lines) >= max_lines:
                    break

    if current and len(lines) < max_lines:
        lines.append(current)

    return lines


# ========= SINGLE LABEL DRAW =========

def draw_single_label(c: canvas.Canvas, row: pd.Series) -> None:
    """
    Draw a single label page on an existing canvas.
    Coordinates: ReportLab origin is bottom-left.
    """

    # Extract fields from the DataFrame row
    sku = str(row.get("SKU_Code", "") or "").strip()
    style = str(row.get("Style", "") or "").strip()
    ean = str(row.get("EAN_Code", "") or "").strip() or sku
    product = str(row.get("Product", "") or "").strip()
    color = str(row.get("Color", "") or "").strip()
    size = str(row.get("Size", "") or "").strip()
    uid = str(row.get("UID", "") or "").strip()

    # DataMatrix payload: STYLE-SKU;UID (only if we have all 3)
    dm_payload = f"{style}-{sku};{uid}" if (style and sku and uid) else ""

    # Default drawing colour for text / lines
    c.setFillColorRGB(0.0, 0.0, 0.0)
    c.setStrokeColorRGB(0.0, 0.0, 0.0)

    dm_img = make_dm_image(dm_payload)

    # ---- TOP DM ----
    dm_size_pt = DM_SIZE_MM * mm
    top_dm_x = DM_LEFT_MM * mm  # close to left edge
    top_dm_y = PAGE_H - TOP_MARGIN_MM * mm - dm_size_pt
    draw_datamatrix(c, dm_img, top_dm_x, top_dm_y, dm_size_pt)

    # ---- TOP SKU (small above big, aligned to right) ----
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

    # Small number baseline above big
    c.setFont(BODY_FONT, BODY_PT)
    small_y_top = big_y_top + 4.0 * mm
    if sku_small:
        c.drawRightString(sku_x, small_y_top, sku_small)

    # ---- UID (centered, below DM block) ----
    uid_y = top_dm_y - UID_GAP_MM * mm
    c.setFont(BODY_FONT, BODY_PT)
    if uid:
        c.drawCentredString(PAGE_W / 2.0, uid_y, uid)

    # ---- BARCODE (wide, lighter) ----
    bc_full_w = PAGE_W - 2.0 * BC_SIDE_MARGIN_MM * mm
    bc_y = uid_y - BARCODE_TOP_GAP_MM * mm
    draw_barcode(c, ean or sku or "000", PAGE_W / 2.0, bc_y, bc_full_w)

    # Human-readable digits under barcode
    hr_y = bc_y - HR_GAP_MM * mm
    human_text = ean or sku or ""
    if human_text:
        c.setFont(BODY_FONT, BODY_PT)
        c.drawCentredString(PAGE_W / 2.0, hr_y, human_text)

    # ---- DIVIDER ----
    divider_y = hr_y - DIVIDER_GAP_MM * mm
    c.setLineWidth(0.4)
    c.line(
        SIDE_MARGIN_MM * mm,
        divider_y,
        PAGE_W - SIDE_MARGIN_MM * mm,
        divider_y,
    )

    # ---- PRODUCT / COLOR / SIZE ----
    text_y = divider_y - TEXT_TOP_GAP_MM * mm
    max_text_width = PAGE_W - 2.0 * SIDE_MARGIN_MM * mm

    c.setFont(BODY_FONT, BODY_PT)

    for line in wrap_text(
        c,
        product,
        max_text_width,
        font_name=BODY_FONT,
        font_size=BODY_PT,
        max_lines=2,
    ):
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
    bottom_dm_x = DM_LEFT_MM * mm
    draw_datamatrix(c, dm_img, bottom_dm_x, bottom_dm_y, dm_size_pt)

    # ---- BOTTOM SKU ----
    center_y_dm_bottom = bottom_dm_y + dm_size_pt / 2.0
    sku_x_bottom = sku_x

    c.setFont(BIG_FONT, BIG_NUM_PT)
    big_y_bottom = center_y_dm_bottom - BIG_NUM_PT * 0.4
    if sku_big:
        c.drawRightString(sku_x_bottom, big_y_bottom, sku_big)
    elif sku:
        c.drawRightString(sku_x_bottom, big_y_bottom, sku)

    c.setFont(BODY_FONT, BODY_PT)
    small_y_bottom = big_y_bottom + 5.0 * mm
    if sku_small:
        c.drawRightString(sku_x_bottom, small_y_bottom, sku_small)


# ========= BATCH PIPELINE =========

def load_uid_excel(path: Path) -> pd.DataFrame:
    """
    Load the UID-level Excel produced by the browser.

    Required columns:
        PO_Number, SKU_Code, EAN_Code, Product, Color, Size, Style, UID

    Extra columns (e.g. Label_Index) are ignored.
    """
    df = pd.read_excel(path, dtype=str)
    required = [
        "PO_Number",
        "SKU_Code",
        "EAN_Code",
        "Product",
        "Color",
        "Size",
        "Style",
        "UID",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Excel: {missing}")
    for col in required:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def process_po_to_pdf(args_tuple):
    """
    Worker function for one PO -> one PDF.
    """
    po_number, po_df, out_dir = args_tuple
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{po_number}.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=(PAGE_W, PAGE_H))

    for _, row in po_df.iterrows():
        draw_single_label(c, row)
        c.showPage()

    c.save()
    return str(pdf_path)


def run_pdf_by_po(
    input_xlsx: Path,
    out_dir: Path,
    processes: int = 0,
    po_filter: Optional[Iterable[str]] = None,
) -> None:
    """
    Main batch entrypoint used by the web server.

    - Reads the UID-level Excel.
    - Optionally filters by PO_Number.
    - Groups labels by PO and renders one PDF per PO.
    """
    df = load_uid_excel(input_xlsx)
    if po_filter:
        allowed = {str(p).strip() for p in po_filter if str(p).strip()}
        if allowed:
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
        for job in jobs:
            result = process_po_to_pdf(job)
            print("Created:", result)


# ========= CLI ENTRY =========

def main() -> None:
    parser = argparse.ArgumentParser(description="UID-level label batch renderer")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to uid_labels.xlsx (UID-level sheet from browser)",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        required=True,
        help="Output directory (PDFs by PO)",
    )
    parser.add_argument(
        "--mode",
        choices=["pdf_by_po"],
        default="pdf_by_po",
        help="Render mode (currently only pdf_by_po)",
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=0,
        help="Number of worker processes (0 => CPU count, 1 => no multiprocessing)",
    )
    parser.add_argument(
        "--po-filter",
        type=str,
        default="",
        help="Optional comma-separated list of PO_Number to include (for testing)",
    )

    args = parser.parse_args()
    input_xlsx = Path(args.input)
    out_dir = Path(args.out_dir)

    if args.mode == "pdf_by_po":
        procs = args.processes or mp.cpu_count()
        po_filter = (
            [p.strip() for p in args.po_filter.split(",") if p.strip()]
            if args.po_filter
            else None
        )
        run_pdf_by_po(input_xlsx, out_dir, processes=procs, po_filter=po_filter)
    else:
        raise SystemExit("Unsupported mode for now.")


if __name__ == "__main__":
    main()
