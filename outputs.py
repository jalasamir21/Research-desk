

 
from __future__ import annotations
 
import io
import smtplib
from email.message import EmailMessage
from xml.sax.saxutils import escape as _xml_escape
 
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable,
)
 
 
def _esc(value) -> str:
    """
    Escapes text before it goes into a reportlab Paragraph.
 
    Paragraph() parses its input as a small XML/HTML-like markup language,
    so any '&', '<', or '>' coming from paper titles, author lists, or
    LLM-generated text (all of which we don't control) can break parsing
    or silently mangle the output. Everything user- or LLM-supplied must
    go through this before being embedded in a Paragraph string.
    """
    return _xml_escape(str(value) if value is not None else "")
 
 
# --------------------------------------------------------------------------
# PDF report generation
# --------------------------------------------------------------------------
def generate_pdf_brief(result: dict) -> bytes:
    """
    Builds a formatted PDF research brief from a pipeline result dict
    (same shape as returned by pipeline.run_pipeline) and returns it as bytes,
    ready for st.download_button.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
 
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "BriefTitle", parent=styles["Title"], fontSize=20, spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        "Meta", parent=styles["Normal"], textColor=colors.grey, fontSize=9, spaceAfter=16,
    )
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14)
    label = ParagraphStyle("Label", parent=styles["Normal"], fontSize=10, leading=14, textColor=colors.HexColor("#333333"))
 
    story = []
 
    story.append(Paragraph(f"Research Brief: {_esc(result['topic'])}", title_style))
    story.append(Paragraph(
        f"Generated {_esc(result['timestamp'])} &middot; {len(result.get('papers', []))} papers analyzed",
        meta_style,
    ))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#DDDDDD")))
 
    story.append(Paragraph("Trends", h2))
    story.append(Paragraph(_esc(result.get("trends", "N/A")), body))
 
    story.append(Paragraph("Gaps", h2))
    story.append(Paragraph(_esc(result.get("gaps", "N/A")), body))
 
    story.append(PageBreak())
    story.append(Paragraph("Papers", h2))
 
    for p in result.get("papers", []):
        story.append(Paragraph(_esc(p["title"]), h3))
        story.append(Paragraph(f"<i>{_esc(p.get('authors', ''))} — {_esc(p.get('published', ''))}</i>", meta_style))
 
        rows = [
            ["Methodology", p.get("methodology", "N/A")],
            ["Datasets", p.get("datasets", "N/A")],
            ["Key results", p.get("key_results", "N/A")],
            ["Limitations", p.get("limitations", "N/A")],
        ]
        table_data = [[Paragraph(f"<b>{_esc(r[0])}</b>", label), Paragraph(_esc(r[1]), body)] for r in rows]
        t = Table(table_data, colWidths=[1.2 * inch, 5.3 * inch])
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#EEEEEE")),
        ]))
        story.append(t)
        paper_url = p.get("url", "")
        story.append(Paragraph(f'<link href="{_esc(paper_url)}">{_esc(paper_url)}</link>', meta_style))
        story.append(Spacer(1, 10))
 
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
 
 
# --------------------------------------------------------------------------
# Email sending
# --------------------------------------------------------------------------
def send_email_brief(
    result: dict,
    pdf_bytes: bytes,
    sender_email: str,
    sender_app_password: str,
    recipient_email: str,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 465,
) -> tuple[bool, str]:
    """
    Emails the PDF research brief to recipient_email as an attachment.
 
    Defaults to Gmail's SMTP-over-SSL endpoint, since "sender app password"
    (collected in the sidebar) is Gmail App Password terminology. Pass a
    different smtp_host/smtp_port for another provider.
 
    Returns (success, message).
    """
    if not sender_email or not sender_app_password:
        return False, "Missing sender email or app password."
    if not recipient_email:
        return False, "Missing recipient email."
 
    # Google displays app passwords as "abcd efgh ijkl mnop" — strip any
    # whitespace in case it was pasted verbatim.
    sender_app_password = sender_app_password.replace(" ", "")
 
    msg = EmailMessage()
    msg["Subject"] = f"Research Brief: {result.get('topic', '')}"
    msg["From"] = sender_email
    msg["To"] = recipient_email
 
    n_papers = len(result.get("papers", []))
    body = (
        f'Your research brief on "{result.get("topic", "")}" is ready.\n\n'
        f"{n_papers} paper(s) analyzed.\n\n"
        f"Trends: {result.get('trends', '')}\n\n"
        f"Gaps: {result.get('gaps', '')}\n\n"
        "See the attached PDF for the full brief."
    )
    msg.set_content(body)
 
    filename = f"brief_{result.get('topic', 'report').replace(' ', '_')}.pdf"
    msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)
 
    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
            server.login(sender_email, sender_app_password)
            server.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        return False, f"SMTP authentication failed — check the sender email and app password: {e}"
    except Exception as e:
        return False, f"Failed to send email: {e}"
 
    return True, f"Emailed brief to {recipient_email}."
 
