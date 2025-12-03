# Copyright (c) Opendatalab. All rights reserved.
import os
import sys
sys.path.append("../")
import time
import argparse

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--enable_ov', action='store_false', default=True, help='enable_ov')
    parser.add_argument('--enable_bf16_det', action='store_false', default=True, help='enable_bf16_det')
    parser.add_argument('--enable_bf16_rec', action='store_false', default=True, help='enable_bf16_rec')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', default="demo/pdfs/ocr.pdf", help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--nstreams', '-n', type=int, default=8, help='Number of ov streams')
    
    return parser.parse_args()

__dir__ = os.path.dirname(os.path.abspath(__file__))

args = get_args()
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
    infer_result = ds.apply(doc_analyze, enable_ov=args.enable_ov, 
                            enable_bf16_det=args.enable_bf16_det, 
                            enable_bf16_rec=args.enable_bf16_rec,
                            nstreams = args.nstreams, 
                            ocr=True)
    pipe_result = infer_result.pipe_ocr_mode(image_writer)
else:
    infer_result = ds.apply(doc_analyze, enable_ov=args.enable_ov, 
                            enable_bf16_det=args.enable_bf16_det, 
                            enable_bf16_rec=args.enable_bf16_rec, 
                            nstreams = args.nstreams,
                            ocr=False)
    pipe_result = infer_result.pipe_txt_mode(image_writer)

### get model inference result
model_inference_result = infer_result.get_infer_res()
    
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
middle_json_content = pipe_result.get_middle_json()
    
### dump middle json
pipe_result.dump_middle_json(md_writer, f'{name_without_extension}_middle.json')
end_time = time.perf_counter()
# print(f"### PostProcess PDF: {(end_time - start_time) * 1000:.2f} ms")
print(f"### Total End2End using time: {(end_time - t0) * 1000:.2f} ms, ",
      f"enable_ov={args.enable_ov}, enable_bf16_det={args.enable_bf16_det}, ",
      f"enable_bf16_rec={args.enable_bf16_rec}, nstreams={args.nstreams}")
    
