
import requests
from pathlib import Path
import argparse
import json

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--url', '-u', type=str, default="http://127.0.0.1:5000/", help='serving url')
    parser.add_argument('--pdf', '-p', type=str, default="/home/sgui/BD/pdf_ocr/MinerU/demo/pdfs/demo1.pdf",
                        help='Filenames of input pdf')
    return parser.parse_args()

args = parse_args()


pdf_path = Path(args.pdf)

session = requests.Session()
session.trust_env = False  # 不读取 HTTP_PROXY/HTTPS_PROXY 等环境变量
proxies = {"http": None, "https": None}

files = {
    "file": (
        pdf_path.name,
        pdf_path.open("rb"),
        "application/pdf",
    )
}

try:
    resp = session.post(args.url, files=files, timeout=300, proxies=proxies)
    resp.raise_for_status()

    # 如果返回是 JSON
    try:
        data = resp.json()
        print(f"JSON 响应：{data.keys()} {len(data['json_raw'])}")
        for line in data['json_raw']:
            # line_data = json.loads(line)
            if line['type'] == 'text' :
                print(f"{line['page_idx']}, {line['type']}, "
                      f"{line['text'] if len(line['text']) < 5 else len(line['text'])}")
            elif line['type'] == 'image' :
                print(f"{line['page_idx']}, {line['type']}, "
                      f"{line['img_caption'][0] if len(line['img_caption'][0]) < 5 else len(line['img_caption'][0])}, "
                      f"{line['img_footnote'] if len(line['img_footnote']) == 0 else len(line['img_footnote'][0])}")
            elif line['type'] == 'table' :
                print(f"{line['page_idx']}, {line['type']}, "
                      f"{line['table_caption'][0] if len(line['table_caption'][0]) < 5 else len(line['table_caption'][0])}, "
                      f"{line['table_footnote'] if len(line['table_footnote']) == 0 else len(line['table_footnote'][0])}")
            elif line['type'] == 'equation' :
                print(f"{line['page_idx']}, {line['type']}, "
                      f"{line['text'] if len(line['text']) == 0 else len(line['text'][0])}, "
                      f"{line['text_format']}")
            else :
                print(f"### {line['page_idx']}, {line['type']}, {line.keys()}")
    except ValueError:
        # 非 JSON 时打印文本
        print("文本响应：", resp.text)

except requests.RequestException as e:
    print(e)