from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import cv2
import os


# ------------------------------------------------------------
# SAVE A NUMPY IMAGE TO TEMP FILE
# ------------------------------------------------------------
def save_temp_image(np_img, filename):
    """
    Saves a BGR or RGB numpy image to a temporary path.
    Returns the file path to be used by reportlab.
    """
    if np_img is None:
        return None

    path = f"/tmp/{filename}"

    # Convert if needed
    if len(np_img.shape) == 3:
        img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    else:
        img = np_img

    cv2.imwrite(path, img)
    return path


# ------------------------------------------------------------
# GENERATE PDF REPORT
# ------------------------------------------------------------
def generate_pdf_report(output_path, original_image, results, aligned_face, logo_path="logo/forailogo.jpg"):
    """
    Creates a full PDF report including original image, region crops,
    and analysis results.

    Parameters:
        output_path: final pdf file path
        original_image: customer's raw face image (numpy)
        results: dict from analyze_face()
        aligned_face: aligned face image (numpy)
        logo_path: FORAE AI logo
    """

    # Prepare PDF
    pdf = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # --------------------------------------------------------
    # LOGO
    # --------------------------------------------------------
    if os.path.exists(logo_path):
        story.append(Image(logo_path, width=120, height=60))
    story.append(Spacer(1, 12))

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------
    title = "<b><font size=18>FORAE AI – Skin Analysis Report</font></b>"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 20))

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------
    orig_path = save_temp_image(original_image, "original.jpg")
    if orig_path:
        story.append(Paragraph("<b>Original Uploaded Image:</b>", styles["Heading2"]))
        story.append(Image(orig_path, width=300, height=300))
        story.append(Spacer(1, 20))

    # --------------------------------------------------------
    # REGION IMAGES
    # --------------------------------------------------------
    story.append(Paragraph("<b>Analyzed Face Regions:</b>", styles["Heading2"]))

    region_imgs = []
    for region_name, data in results.items():
        crop_img = data["crop"]
        crop_path = save_temp_image(crop_img, f"{region_name}.jpg")

        if crop_path:
            region_imgs.append([Paragraph(region_name.capitalize(), styles["Normal"]),
                                Image(crop_path, width=150, height=150)])

    table = Table(region_imgs, colWidths=[2.2 * inch, 2.2 * inch])
    table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOX', (0,0), (-1,-1), 1, colors.gray),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.lightgrey)
    ]))
    story.append(table)
    story.append(Spacer(1, 25))


    # --------------------------------------------------------
    # ANALYSIS SUMMARY (ACNE, PORES, WRINKLES, REDNESS, OILINESS)
    # --------------------------------------------------------
    story.append(Paragraph("<b>Skin Issues Summary:</b>", styles["Heading2"]))

    for region_name, data in results.items():
        story.append(Paragraph(f"<b>{region_name.capitalize()}:</b>", styles["Heading3"]))

        acne = data["acne"]["score"]
        pores = data["pores"]["score"]
        wrinkles = data["wrinkles"]["score"]
        redness = data["redness"]["score"]
        oiliness = data["oiliness"]["score"]

        summary = f"""
        <b>Acne:</b> {acne} points<br/>
        <b>Pores:</b> {pores} detected<br/>
        <b>Wrinkles:</b> {wrinkles} detected<br/>
        <b>Redness:</b> {redness} intensity<br/>
        <b>Oiliness:</b> {oiliness} shine score<br/>
        """
        story.append(Paragraph(summary, styles["Normal"]))
        story.append(Spacer(1, 12))

    story.append(Spacer(1, 20))

    # --------------------------------------------------------
    # SKIN TONE SUMMARY
    # --------------------------------------------------------
    story.append(Paragraph("<b>Skin Tone Analysis:</b>", styles["Heading2"]))

    tone = results["forehead"]["skin_tone"]  # using forehead region for tone
    depth = tone["depth"]
    undertone = tone["undertone"]

    story.append(Paragraph(f"<b>Depth:</b> {depth}", styles["Normal"]))
    story.append(Paragraph(f"<b>Undertone:</b> {undertone}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------
    story.append(Paragraph("<b>AI Recommendations:</b>", styles["Heading2"]))

    rec = """
    • Use a gentle cleanser twice daily<br/>
    • Apply non-comedogenic moisturizer<br/>
    • Use sunscreen (SPF 30+) daily<br/>
    • For acne: salicylic acid or benzoyl peroxide<br/>
    • For pores: niacinamide + clay mask weekly<br/>
    • For redness: azelaic acid or centella<br/>
    • For wrinkles: retinol at night + vitamin C<br/>
    """
    story.append(Paragraph(rec, styles["Normal"]))

    story.append(Spacer(1, 20))

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------
    footer = "<font size=10>Report generated by FORAE AI © 2025</font>"
    story.append(Paragraph(footer, styles["Normal"]))

    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------
    pdf.build(story)

    return output_path

