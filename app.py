from flask import Flask, request, jsonify
from PIL import Image
import io
import sys
import os
import uuid
import time
import argparse
import numpy as np
from os import PathLike
from pathlib import Path
from magic_pdf.model.doc_analyze_by_custom_model import init_models, doc_analyze_direct

app = Flask(__name__)

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--enable_ov', '-e', action='store_true', default=False, help='enable_ov')
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
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', default="demo/pdfs/demo1.pdf",
                        help='Filenames of input pdfs')
    parser.add_argument('--nstreams', '-n', type=int, default=8, help='Number of ov streams')
    parser.add_argument('--app', '-p', action='store_true', default=False,
                        help='True for app, False for serving')
    # fmt: on
    return parser.parse_args()

args = parse_args()
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

class PDF_Instance :
    def __init__(self, args) :
        if args.all is not None :
            args.layout_type = args.all
            args.mfd_type = args.all
            args.mfr_enc_type = args.all
            args.mfr_dec_type = args.all
            args.ocr_det_type = args.all
            args.ocr_rec_type = args.all
            args.table_type = args.all
            args.lang_type = args.all
            args.page_type = args.all
        self.enable_ov = args.enable_ov
        self.layout_type = args.layout_type
        self.mfd_type = args.mfd_type
        self.mfr_enc_type = args.mfr_enc_type
        self.mfr_dec_type = args.mfr_dec_type
        self.ocr_det_type = args.ocr_det_type
        self.ocr_rec_type = args.ocr_rec_type
        self.table_type = args.table_type
        self.lang_type = args.lang_type
        self.page_type = args.page_type
        self.nstreams = args.nstreams
        self.return_md = True
        self.return_json = True        
        self.pdf_model = init_models(args.enable_ov, args.layout_type, args.mfd_type, args.mfr_enc_type,
                        args.mfr_dec_type, args.ocr_det_type, args.ocr_rec_type, args.table_type,
                        args.lang_type, args.page_type, args.nstreams, True)

    def process_pdf(self, pdf_raw: bytes) :
        return doc_analyze_direct(pdf_raw, self.pdf_model, self.enable_ov, self.layout_type, self.mfd_type,
                                  self.mfr_enc_type, self.mfr_dec_type, self.ocr_det_type,
                                  self.ocr_rec_type, self.table_type, self.lang_type, self.page_type,
                                  self.nstreams, self.return_md, self.return_json)

pdf_instance = PDF_Instance(args)

def download_file(
    url: PathLike,
    filename: PathLike = None,
    directory: PathLike = None,
    show_progress: bool = True,
) -> PathLike:
    """
    Download a file from a url and save it to the local filesystem. The file is saved to the
    current directory by default, or to `directory` if specified. If a filename is not given,
    the filename of the URL will be used.

    :param url: URL that points to the file to download
    :param filename: Name of the local file to save. Should point to the name of the file only,
                     not the full path. If None the filename from the url will be used
    :param directory: Directory to save the file to. Will be created if it doesn't exist
                      If None the file will be saved to the current working directory
    :param show_progress: If True, show an TQDM ProgressBar
    :param silent: If True, do not print a message if the file already exists
    :param timeout: Number of seconds before cancelling the connection attempt
    :return: path to downloaded file
    """
    # from tqdm import tqdm
    import requests
    import urllib.parse

    filename = filename or Path(urllib.parse.urlparse(url).path).name
    chunk_size = 16384  # make chunks bigger so that not too many updates are triggered for Jupyter front-end

    filename = Path(filename)
    if len(filename.parts) > 1:
        raise ValueError(
            "`filename` should refer to the name of the file, excluding the directory. "
            "Use the `directory` parameter to specify a target directory for the downloaded file."
        )

    filepath = Path(directory) / filename if directory is not None else filename
    if filepath.exists():
        return filepath.resolve()

    # create the directory if it does not exist, and add the directory to the filename
    if directory is not None:
        Path(directory).mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url=url, headers={"User-agent": "Mozilla/5.0"}, stream=True)
        response.raise_for_status()
    except (
        requests.exceptions.HTTPError
    ) as error:  # For error associated with not-200 codes. Will output something like: "404 Client Error: Not Found for url: {url}"
        raise Exception(error) from None
    except requests.exceptions.Timeout:
        raise Exception(
            "Connection timed out. If you access the internet through a proxy server, please "
            "make sure the proxy is set in the shell from where you launched Jupyter."
        ) from None
    except requests.exceptions.RequestException as error:
        raise Exception(f"File downloading failed with error: {error}") from None

    # download the file if it does not exist
    filesize = int(response.headers.get("Content-length", 0))
    if not filepath.exists():
        with open(filepath, "wb") as file_object:
            for chunk in response.iter_content(chunk_size):
                file_object.write(chunk)
    else:
        print(f"'{filepath}' already exists.")

    response.close()

    return filepath.resolve()

def load_pdf_file(file_path):
    with open(file_path, 'rb') as f:
        f.seek(0)
        return f.read()
            
@app.route('/', methods=['POST'])
def pdf_process():
    pdf_raw = None
    if request.is_json:
        json_data = request.get_json()
        if json_data:
            if 'url' in json_data:
                random_uuid = uuid.uuid4()
                filename = f"{random_uuid}.pdf"
                filename = download_file(
                    url=json_data['url'],
                    filename=filename,
                    directory='/tmp'
                )
                pdf_raw = load_pdf_file(filename)
                os.remove(filename)
            elif 'pdf_raw' in json_data:
                pdf_raw = json_data['pdf_raw'],
            else :
                return jsonify({'error': 'Unsupported JSON format. Expected {"url": "address"} or {"pdf_raw": "data"}'}), 400
        else:
            return jsonify({'error': 'Invalid JSON format. Expected {"url": "address"} or {"pdf_raw": "data"}'}), 400
    elif 'filename' in request.form:
        if os.path.exists(request.form['filename']):
            # load pdf file
            pdf_raw = load_pdf_file(request.form['filename'])
        else:
            return jsonify({'error': 'Failed to open image file'}), 400
    elif 'file' in request.files:       
        pdf_file = request.files['file']
        pdf_raw = pdf_file.read()
    elif 'url' in request.form:
        random_uuid = uuid.uuid4()
        filename = f"{random_uuid}.pdf"
        filename = download_file(
            url=request.form['url'],
            filename=filename,
            directory='/tmp'
        )
        if os.path.exists(filename):
            pdf_raw = load_pdf_file(filename)
            os.remove(filename)
        else:
            return jsonify({'error': 'Failed to download PDF from URL'}), 400
    else :
        return jsonify({'error': 'No PDF uploaded or filename provided'}), 400
    if pdf_raw is None :
        return jsonify({'error': 'PDF data is invalid'}), 400
    start_time = time.perf_counter()
    (md_raw, json_raw) = pdf_instance.process_pdf(pdf_raw)
    end_time = time.perf_counter()
    print(f"Processing:  {end_time-start_time:.3f}")
    return jsonify({'json_raw': json_raw})

if __name__ == '__main__':
    if args.app:
        if args.input is None :
            print(f"app mode need set input")
            exit(0)
        elif isinstance(args.input, str) :
            args.input = [args.input]
        for input_name in args.input:
            if os.path.isdir(input_name) :  
                for root, dirs, files in os.walk(input_name):
                    for f in files:
                        if f.lower().endswith("pdf"):
                            full_path = os.path.join(root, f)
                            pdf_raw = load_pdf_file(full_path)
                            (md_raw, json_raw) = pdf_instance.process_pdf(pdf_raw)
            elif os.path.isfile(input_name):
                pdf_raw = load_pdf_file(input_name)
                (md_raw, json_raw) = pdf_instance.process_pdf(pdf_raw)
            else :
                print(f"app mode need set input")
    else :
        app.run(host='0.0.0.0', port=5000)