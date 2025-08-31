import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import pandas as pd

def generate_report():
    # Rapor dosya yolu
    report_path = "results/model_report.pdf"
    os.makedirs("results", exist_ok=True)

    # PDF ayarları
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Başlık
    elements.append(Paragraph("Stock Price Prediction Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    # Evaluation sonuçlarını ekle
    metrics_file = "results/evaluation_metrics.csv"
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        elements.append(Paragraph("Evaluation Metrics", styles["Heading2"]))
        data = [df.columns.tolist()] + df.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME', (0,0),(-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0),(-1,0), 12),
            ('BACKGROUND',(0,1),(-1,-1),colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

    # Training loss grafiği ekle
    loss_img = "results/training_loss.png"
    if os.path.exists(loss_img):
        elements.append(Paragraph("Training Loss Over Epochs", styles["Heading2"]))
        elements.append(Image(loss_img, width=400, height=250))
        elements.append(Spacer(1, 20))

    # Future prediction grafiği ekle
    future_img = "results/future_predictions.png"
    if os.path.exists(future_img):
        elements.append(Paragraph("Future Price Predictions", styles["Heading2"]))
        elements.append(Image(future_img, width=400, height=250))
        elements.append(Spacer(1, 20))

    # Kapanış
    elements.append(Paragraph("Report generated successfully.", styles["Normal"]))

    # PDF oluştur
    doc.build(elements)
    print(f"Report saved at {report_path}")


if __name__ == "__main__":
    generate_report()
