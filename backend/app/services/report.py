"""PDF report for a single prediction (ReportLab)."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .. import config

NAVY = colors.HexColor("#0a2352")
BLUE = colors.HexColor("#0b84ff")
LIGHT = colors.HexColor("#eaf3ff")
CLASS_COLORS = ["#16a34a", "#d99a06", "#ea6a0c", "#dc2626", "#7c3aed"]


def _fmt_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y, %I:%M %p UTC")
    except Exception:
        return iso


def build_pdf(pred: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=f"DR Vision AI – Prediction #{pred['id']}",
                            author="DR Vision AI")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=NAVY, fontSize=20, alignment=0, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=ss["Normal"], textColor=colors.HexColor("#4f6a93"), fontSize=9)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=12, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=10, leading=14, textColor=colors.HexColor("#1b3358"))
    small = ParagraphStyle("small", parent=body, fontSize=8.5, textColor=colors.HexColor("#4f6a93"))

    cls = pred["class_index"]
    story = [
        Paragraph("DR Vision AI", h1),
        Paragraph("Diabetic Retinopathy Severity Classification – Prediction Report", sub),
        Spacer(1, 6 * mm),
    ]

    if pred["mode"] == "demo":
        warn = Table([[Paragraph(
            "<b>DEMO MODE:</b> no trained model was available, so this result comes from a simple image-statistics "
            "heuristic and has no diagnostic meaning.", body)]], colWidths=[174 * mm])
        warn.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4d6")),
                                  ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#e0b341")),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        story += [warn, Spacer(1, 4 * mm)]

    meta = [
        ["Prediction ID", f"#{pred['id']}"],
        ["File", escape(str(pred["filename"]))],
        ["Date", _fmt_date(pred["created_at"])],
        ["Model", config.MODEL_BACKBONES],
        ["Classifier", config.MODEL_CLASSIFIER],
        ["Model version", escape(str(pred.get("model_version", "unknown")))],
        ["Processing time", f"{pred['processing_time']:.2f} seconds"],
    ]
    t = Table(meta, colWidths=[40 * mm, 134 * mm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9.5), ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
                           ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                           ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT]),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [t, Paragraph("Predicted severity", h2)]

    result = Table([[Paragraph(f"<font color='{CLASS_COLORS[cls]}'><b>{pred['class_name']}</b></font> "
                               f"&nbsp;(Class {cls})", ParagraphStyle("r", parent=body, fontSize=15)),
                     Paragraph(f"Top class score<br/><font size=15 color='#0b84ff'><b>{pred['confidence'] * 100:.2f}%</b>"
                               "</font>", body)]], colWidths=[110 * mm, 64 * mm])
    result.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#b9d4f2")),
                                ("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                                ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    story += [result, Paragraph("Class scores", h2)]

    rows = [["Class", "Model score"]] + [[f"{n} ({i})", f"{p * 100:.2f}%"] for i, (n, p) in
                                          enumerate(zip(config.CLASS_NAMES, pred["probabilities"]))]
    pt = Table(rows, colWidths=[110 * mm, 64 * mm])
    pt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9dcf3")),
                            ("FONTNAME", (0, cls + 1), (-1, cls + 1), "Helvetica-Bold")]))
    story += [pt, Paragraph(escape(str(pred.get("score_description", "Classifier output scores; not probabilities of actual disease."))), small),
              Paragraph("Images", h2)]

    imgs, caps = [], []
    for key, cap in (("image_file", "Original"), ("heatmap_file", "Heatmap"), ("overlay_file", "Overlay")):
        fname = pred.get(key)
        folder = config.UPLOAD_DIR if key == "image_file" else config.HEATMAP_DIR
        p = Path(folder) / fname if fname else None
        if p and p.exists():
            imgs.append(Image(str(p), width=52 * mm, height=52 * mm))
            caps.append(Paragraph(cap, small))
    if imgs:
        it = Table([imgs, caps], colWidths=[58 * mm] * len(imgs))
        it.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(it)

    story += [Paragraph("About this prediction", h2),
              Paragraph(f"The model predicted <b>{escape(str(pred['class_name']))}</b> (Class {cls}) with "
                        f"{pred['confidence'] * 100:.2f}% confidence. This indicates {config.CLASS_EXPLANATIONS[cls]}.",
                        body),
              Spacer(1, 5 * mm),
              Paragraph("<b>Disclaimer:</b> this application is for educational and research purposes only. "
                        "It is not a substitute for professional medical advice, diagnosis or treatment.", small)]
    doc.build(story)
    return buf.getvalue()
