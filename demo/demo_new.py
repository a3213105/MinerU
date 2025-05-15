# Copyright (c) Opendatalab. All rights reserved.
import os
import sys
sys.path.append("../")
import time

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

enable_ov = True
enable_bf16_det = True
enable_bf16_rec = True
nstreams = 8
# args
__dir__ = os.path.dirname(os.path.abspath(__file__))
# pdf_file_name = os.path.join(__dir__, "pdfs", "demo1.pdf")  # replace with the real pdf path
pdf_file_name = os.path.join(__dir__, "pdfs", "ocr.pdf")  # replace with the real pdf path
name_without_extension = os.path.basename(pdf_file_name).split('.')[0]

# prepare env
local_image_dir = os.path.join(__dir__, "output", name_without_extension, "images")
local_md_dir = os.path.join(__dir__, "output", name_without_extension)
image_dir = str(os.path.basename(local_image_dir))
os.makedirs(local_image_dir, exist_ok=True)

image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

# read bytes
reader1 = FileBasedDataReader("")
pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content

# proc
## Create Dataset Instance
# print("### Create Dataset Instance...")
t0 = time.perf_counter()
start_time = t0
ds = PymuDocDataset(pdf_bytes)
# end_time = time.perf_counter()
# print(f"### Create Dataset: {(end_time - start_time) * 1000:.2f} ms")
## inference
if ds.classify() == SupportedPdfParseMethod.OCR:
    # print("### ds.apply(doc_analyze, ocr=True)...")
    # start_time = time.perf_counter()
    infer_result = ds.apply(doc_analyze, enable_ov=enable_ov, 
                            enable_bf16_det=enable_bf16_det, 
                            enable_bf16_rec=enable_bf16_rec,
                            nstreams = nstreams, 
                            ocr=True)
    # end_time = time.perf_counter()
    # print(f"### doc_analyze, ocr=True: {(end_time - start_time) * 1000:.2f} ms")

    # print("### infer_result.pipe_ocr_mode...")
    ## pipeline
    # start_time = time.perf_counter()
    pipe_result = infer_result.pipe_ocr_mode(image_writer)
    # end_time = time.perf_counter()
    # print(f"### pipe_ocr_mode: {(end_time - start_time) * 1000:.2f} ms")

else:
    # print("### ds.apply(doc_analyze, ocr=False)...")
    # start_time = time.perf_counter()
    infer_result = ds.apply(doc_analyze, enable_ov=enable_ov, 
                            enable_bf16_det=enable_bf16_det, 
                            enable_bf16_rec=enable_bf16_rec, 
                            nstreams = nstreams,
                            ocr=False)
    # end_time = time.perf_counter()
    # print(f"### doc_analyze, ocr=False: {(end_time - start_time) * 1000:.2f} ms")

    # print("### infer_result.pipe_txt_mode...")
    ## pipeline
    # start_time = time.perf_counter()
    pipe_result = infer_result.pipe_txt_mode(image_writer)
    # end_time = time.perf_counter()
    # print(f"### pipe_txt_mode using time: {(end_time - start_time) * 1000:.2f} ms")

# print("### PostProcess inference result...")
### get model inference result
# start_time = time.perf_counter()
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
      f"enable_ov={enable_ov}, enable_bf16_det={enable_bf16_det}, ",
      f"enable_bf16_rec={enable_bf16_rec}, nstreams={nstreams}")
    