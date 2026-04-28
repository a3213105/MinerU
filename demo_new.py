# Copyright (c) Opendatalab. All rights reserved.
import os
import sys
sys.path.append("./")
import time
import argparse

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--disable_ov', '-o', action='store_true', default=False, help='disable_ov')
    parser.add_argument('--layout_type', type=str, default="bf16", help='layout detection infer type')
    parser.add_argument('--mfd_type', type=str, default="bf16", help='formula detection infer type')
    parser.add_argument('--mfr_enc_type', type=str, default="bf16", help='formula recognition enc infer type')
    parser.add_argument('--mfr_dec_type', type=str, default="bf16", help='formula recognition dec infer type')
    parser.add_argument('--ocr_det_type', type=str, default="bf16", help='ocr detection infer type')
    parser.add_argument('--ocr_rec_type', type=str, default="bf16", help='ocr recognition infer type')
    parser.add_argument('--table_type', type=str, default="bf16", help='table infer type')
    parser.add_argument('--lang_type', type=str, default="bf16", help='language detection infer type')
    parser.add_argument('--page_type', type=str, default="bf16", help='page layout infer type')
    parser.add_argument('--all', '-a', type=str, default=None, help='set all infer type')
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', default="demo/pdfs/demo1.pdf",
    #                     help='Filenames of input pdfs')
    parser.add_argument('--input', '-i', default="demo/pdfs/demo1.pdf",
                        help='Filenames of input pdfs')
    parser.add_argument('--nstreams', '-n', type=int, default=8, help='Number of ov streams')
    
    return parser.parse_args()

__dir__ = os.path.dirname(os.path.abspath(__file__))

args = get_args()

if args.all is not None :
    args.all = args.all.lower()
    args.layout_type = args.all
    args.mfd_type = args.all
    args.mfr_enc_type = args.all
    args.mfr_dec_type = args.all
    args.ocr_det_type = args.all
    args.ocr_rec_type = args.all
    args.table_type = args.all
    args.lang_type = args.all
    args.page_type = args.all
else :
    args.layout_type = args.layout_type.lower()
    args.mfd_type = args.mfd_type.lower()
    args.mfr_enc_type = args.mfr_enc_type.lower()
    args.mfr_dec_type = args.mfr_dec_type.lower()
    args.ocr_det_type = args.ocr_det_type.lower()
    args.ocr_rec_type = args.ocr_rec_type.lower()
    args.table_type = args.table_type.lower()
    args.lang_type = args.lang_type.lower()
    args.page_type = args.page_type.lower()


# args
name_without_extension = os.path.basename(args.input).split('.')[0]

# prepare env
local_image_dir = os.path.join(__dir__, "output", name_without_extension, "images")
local_md_dir = os.path.join(__dir__, "output", name_without_extension)
image_dir = str(os.path.basename(local_image_dir))
os.makedirs(local_image_dir, exist_ok=True)

image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

# read bytes
reader1 = FileBasedDataReader("")
pdf_bytes = reader1.read(args.input)  # read the pdf content

# proc
## Create Dataset Instance
t0 = time.perf_counter()
start_time = t0
ds = PymuDocDataset(pdf_bytes)
## inference
if ds.classify() == SupportedPdfParseMethod.OCR:
    infer_result = ds.apply(doc_analyze,
                            enable_ov=(not args.disable_ov),
                            Layout_infer_type=args.layout_type,
                            MFD_infer_type=args.mfd_type,
                            MFR_enc_infer_type=args.mfr_enc_type,
                            MFR_dec_infer_type=args.mfr_dec_type,
                            OCR_det_infer_type=args.ocr_det_type,
                            OCR_rec_infer_type=args.ocr_rec_type,
                            Table_infer_type=args.table_type,
                            Lang_infer_type=args.lang_type,
                            Page_infer_type=args.page_type,
                            nstreams = args.nstreams,
                            ocr=True)
    pipe_result = infer_result.pipe_ocr_mode(image_writer)
else:
    infer_result = ds.apply(doc_analyze,
                            enable_ov=(not args.disable_ov),
                            Layout_infer_type=args.layout_type,
                            MFD_infer_type=args.mfd_type,
                            MFR_enc_infer_type=args.mfr_enc_type,
                            MFR_dec_infer_type=args.mfr_dec_type,
                            OCR_det_infer_type=args.ocr_det_type,
                            OCR_rec_infer_type=args.ocr_rec_type,
                            Table_infer_type=args.table_type,
                            Lang_infer_type=args.lang_type,
                            Page_infer_type=args.page_type,
                            nstreams = args.nstreams,
                            ocr=False)
    pipe_result = infer_result.pipe_txt_mode(image_writer)

# print(f"pipe_result={pipe_result}")

# ### get model inference result
# model_inference_result = infer_result.get_infer_res()
# print(f"model_inference_result={model_inference_result}")

### draw layout result on each page
pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_extension}_layout.pdf"))

### draw spans result on each page
pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_extension}_spans.pdf"))

### get markdown content
md_content = pipe_result.get_markdown(image_dir)

### dump markdown
pipe_result.dump_md(md_writer, f"{name_without_extension}.md", image_dir)

### get content list content
content_list_content = pipe_result.get_content_list(image_dir)

### dump content list
pipe_result.dump_content_list(md_writer, f"{name_without_extension}_content_list.json", image_dir)

### get middle json
# middle_json_content = pipe_result.get_middle_json()

### dump middle json
pipe_result.dump_middle_json(md_writer, f'{name_without_extension}_middle.json')
end_time = time.perf_counter()
# print(f"### PostProcess PDF: {(end_time - start_time) * 1000:.2f} ms")
print(f"### Total End2End using time: {(end_time - t0) * 1000:.2f} ms, ",
      f"enable_ov={(not args.disable_ov)}, Layout_infer_type={args.layout_type}, ",
      f"MFD_infer_type={args.mfd_type}, MFR_enc_infer_type={args.mfr_enc_type}, ",
      f"MFR_dec_infer_type={args.mfr_dec_type}, OCR_det_infer_type={args.ocr_det_type}, ",
      f"OCR_rec_infer_type={args.ocr_rec_type}, Table_infer_type={args.table_type}, ",
      f"Lang_infer_type={args.lang_type}, Page_infer_type={args.page_type}, ",
      f"nstreams={args.nstreams}")