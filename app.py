from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import numpy as np
import cv2
import os
from analysis.skin_analysis import analyze_face
from utils.pdf_generator import generate_pdf_report

app = FastAPI()

@app.post("/analyze-face")
async def analyze_face_api(file: UploadFile = File(...)):

    # Read image uploaded by customer
    img_bytes = await file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    # Run AI analysis
    aligned, regions, results = analyze_face(image)

    # Filepath for PDF report
    output_pdf = "reports/analysis_report.pdf"

    os.makedirs("reports", exist_ok=True)

    # Generate PDF
    generate_pdf_report(
        output_path=output_pdf,
        original_image=image,
        results=results,
        aligned_face=aligned
    )

    # Return the report as downloadable file
    return FileResponse(
        output_pdf,
        media_type="application/pdf",
        filename="ForaeAI_Skin_Report.pdf"
    )


@app.get("/")
def home():
    return {"message": "FORAE AI backend running!"}
