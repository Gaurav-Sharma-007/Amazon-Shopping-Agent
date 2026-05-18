import boto3
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO

# Create Bedrock client
client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

# Read original PDF
reader = PdfReader("NASA-Test.pdf")

# Create new PDF with first 2 pages
writer = PdfWriter()

for i in range(min(2, len(reader.pages))):
    writer.add_page(reader.pages[i])

# Store smaller PDF in memory
pdf_buffer = BytesIO()
writer.write(pdf_buffer)

# Convert to bytes
pdf_bytes = pdf_buffer.getvalue()

# Send to Bedrock
response = client.converse(
    modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "document": {
                        "format": "pdf",
                        "name": "NASA-Test-First-2-Pages",
                        "source": {
                            "bytes": pdf_bytes
                        }
                    }
                },
                {
                    "text": "Summarize only these pages."
                }
            ]
        }
    ]
)

print(
    response["output"]["message"]["content"][0]["text"]
)