from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.document_converter import PdfFormatOption
from pathlib import Path

input_dir = Path.home() / "jai-archive/ocr"
output_dir = Path.home() / "jai-archive/markdown"
output_dir.mkdir(exist_ok=True)

# GTX 1070 (sm_61/Pascal) not supported by PyTorch on Python 3.14 — force CPU.
pdf_opts = PdfPipelineOptions()
pdf_opts.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
)

pdfs = list(input_dir.rglob("*.pdf"))
if not pdfs:
    print("No PDFs found in ocr/ folder")
    exit()

print(f"Found {len(pdfs)} documents, checking for new ones...")
new_pdfs = []
for pdf in pdfs:
    relative = pdf.relative_to(input_dir)
    flat_name = str(relative).replace("/", "_").replace("\\", "_")
    out_file = output_dir / (Path(flat_name).stem + ".md")
    if out_file.exists():
        continue
    new_pdfs.append(pdf)

if not new_pdfs:
    print("Nothing new to convert.")
    exit()

print(f"Converting {len(new_pdfs)} new documents...")
for pdf in new_pdfs:
    relative = pdf.relative_to(input_dir)
    flat_name = str(relative).replace("/", "_").replace("\\", "_")
    out_file = output_dir / (Path(flat_name).stem + ".md")

    print(f"  Converting: {relative}")
    try:
        result = converter.convert(str(pdf))
        markdown = result.document.export_to_markdown()
        out_file.write_text(markdown)
        print(f"  Saved: {out_file.name}")
    except Exception as e:
        print(f"  SKIPPED: {relative} — {e}")

print(f"Conversion complete. Output in {output_dir}")
