import os
import time
from typing import List, Tuple
from PIL import Image
from loguru import logger
import pypdfium2 as pdfium

from .model_init import MineruPipelineModel
from mineru.utils.config_reader import get_device
from mineru.utils.enum_class import ImageType
from mineru.utils.pdf_classify import classify
from mineru.utils.pdf_image_tools import load_images_from_pdf, load_image_from_pdf
from mineru.utils.model_utils import get_vram, clean_memory
from mineru.utils.pdf_page_id import get_end_page_id
import gc

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 让mps可以fallback
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新

class ModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    def clear_cache(self):
        keys_to_delete = []
        for k in list(self._models.keys()):
            keys_to_delete.append(k)

        for k in keys_to_delete:
            model = self._models.pop(k, None)
            if model is not None:
                del model
        gc.collect()


    def get_model(self, enable_cache, lang=None, formula_enable=None, table_enable=None, **kwargs):
        key = (lang, formula_enable, table_enable)
        if key in self._models:
            return self._models[key]
        if not enable_cache:
            self.clear_cache()
        self._models[key] = custom_model_init(lang=lang, formula_enable=formula_enable,
                table_enable=table_enable, enable_cache=enable_cache, **kwargs)
        return self._models[key]


def custom_model_init(
    lang=None,
    formula_enable=True,
    table_enable=True,
    **kwargs
):
    model_init_start = time.time()
    # 从配置文件读取model-dir和device
    # device = get_device()
    device = 'cpu'
    formula_config = {"enable": formula_enable}
    table_config = {"enable": table_enable}

    model_input = {
        'device': device,
        'table_config': table_config,
        'formula_config': formula_config,
        'lang': lang,
    }
    model_input.update(kwargs)

    custom_model = MineruPipelineModel(**model_input)

    model_init_cost = time.time() - model_init_start
    logger.info(f'model init cost: {model_init_cost}')

    return custom_model


def doc_analyze(batch_model, pdf_bytes_list, lang_list, enable_cache, enable_ov, Layout_infer_type,
                MFD_infer_type, MFR_enc_infer_type, MFR_dec_infer_type, OCR_det_infer_type, OCR_rec_infer_type,
                wired_table_type, WirelessTable_type, img_orientation_cls_type, table_cls_type, nstreams,
                parse_method: str = 'auto', formula_enable=True, table_enable=True, start_page_id=0, end_page_id=None,):
    """
    适当调大MIN_BATCH_INFERENCE_SIZE可以提高性能，更大的 MIN_BATCH_INFERENCE_SIZE会消耗更多内存，
    可通过环境变量MINERU_MIN_BATCH_INFERENCE_SIZE设置，默认值为384。
    """
    min_batch_inference_size = int(os.environ.get('MINERU_MIN_BATCH_INFERENCE_SIZE', 384))

    # 收集所有页面信息
    all_pages_info = []  # 存储(dataset_index, page_index, img, ocr, lang, width, height)

    all_image_lists = []
    all_pdf_docs = []
    ocr_enabled_list = []
    load_images_start = time.time()
    for pdf_idx, pdf_bytes in enumerate(pdf_bytes_list):
        # 确定OCR设置
        _ocr_enable = False
        if parse_method == 'auto':
            if classify(pdf_bytes) == 'ocr':
                _ocr_enable = True
        elif parse_method == 'ocr':
            _ocr_enable = True

        ocr_enabled_list.append(_ocr_enable)
        _lang = lang_list[pdf_idx]

        # 收集每个数据集中的页面
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL, start_page_id=start_page_id, end_page_id=end_page_id,)
        all_image_lists.append(images_list)
        all_pdf_docs.append(pdf_doc)
        for page_idx in range(len(images_list)):
            img_dict = images_list[page_idx]
            all_pages_info.append((
                pdf_idx, page_idx,
                img_dict['img_pil'], _ocr_enable, _lang,
            ))
    load_images_time = round(time.time() - load_images_start, 2)
    logger.info(f"load images cost: {load_images_time}, speed: {round(len(all_pages_info) / load_images_time, 3)} images/s")

    # 准备批处理
    images_with_extra_info = [(info[2], info[3], info[4]) for info in all_pages_info]
    batch_size = min_batch_inference_size
    batch_images = [
        images_with_extra_info[i:i + batch_size]
        for i in range(0, len(images_with_extra_info), batch_size)
    ]
    # batch_images = [[(info[2], info[3], info[4])] for info in all_pages_info]

    # 执行批处理
    results = []
    processed_images_count = 0
    infer_start = time.time()
    for index, batch_image in enumerate(batch_images):
        processed_images_count += len(batch_image)
        # logger.info(
        #     f'Batch {index + 1}/{len(batch_images)}: '
        #     f'{processed_images_count} pages/{len(images_with_extra_info)} pages'
        # )
        batch_results = batch_image_analyze(batch_model, batch_image)
        results.extend(batch_results)

    infer_time = round(time.time() - infer_start, 2)
    logger.debug(f"infer finished, cost: {infer_time}, speed: {round(len(results) / infer_time, 3)} page/s")

    # 构建返回结果
    infer_results = []

    for _ in range(len(pdf_bytes_list)):
        infer_results.append([])

    for i, page_info in enumerate(all_pages_info):
        pdf_idx, page_idx, pil_img, _, _ = page_info
        result = results[i]

        page_info_dict = {'page_no': page_idx, 'width': pil_img.width, 'height': pil_img.height}
        page_dict = {'layout_dets': result, 'page_info': page_info_dict}

        infer_results[pdf_idx].append(page_dict)

    return infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list


def doc_analyze_1by1(batch_model, pdf_bytes, lang, enable_cache, enable_ov, Layout_infer_type,
                MFD_infer_type, MFR_enc_infer_type, MFR_dec_infer_type, OCR_det_infer_type, OCR_rec_infer_type,
                wired_table_type, WirelessTable_type, img_orientation_cls_type, table_cls_type, nstreams,
                parse_method: str = 'auto', formula_enable=True, table_enable=True, start_page_id=0, end_page_id=None,):
    # 确定OCR设置
    _ocr_enable = False

    if parse_method == 'auto':
        if classify(pdf_bytes) == 'ocr':
            _ocr_enable = True
    elif parse_method == 'ocr':
        _ocr_enable = True

    _lang = lang

    pdf_doc = pdfium.PdfDocument(pdf_bytes)

    end_page_id = get_end_page_id(end_page_id, len(pdf_doc)) + 1
    processed_images_count = 0
    results = []
    image_lists = []
    for page_idx in range(start_page_id, end_page_id):
        page_image_info = load_image_from_pdf(pdf_bytes, pdf_doc, image_type=ImageType.PIL, start_page_id=page_idx, end_page_id=page_idx,)[0]
        image_lists.append(page_image_info)
        page_image = page_image_info['img_pil']
        batch_image = [(page_image, _ocr_enable, _lang)]

        # 执行批处理
        processed_images_count += 1
        batch_results = batch_image_analyze(batch_model, batch_image)
        # 构建返回结果
        page_info_dict = {'page_no': page_idx, 'width': page_image.width, 'height': page_image.height}
        page_dict = {'layout_dets': batch_results[0], 'page_info': page_info_dict}
        results.append(page_dict)
    return results, image_lists, pdf_doc, _lang, _ocr_enable


def get_batch_info():
    device = get_device()

    if str(device).startswith('npu'):
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                torch_npu.npu.set_compile_mode(jit_compile=False)
        except Exception as e:
            raise RuntimeError(
                "NPU is selected as device, but torch_npu is not available. "
                "Please ensure that the torch_npu package is installed correctly."
            ) from e

    gpu_memory = get_vram(device)
    if gpu_memory >= 16:
        batch_ratio = 16
    elif gpu_memory >= 12:
        batch_ratio = 8
    elif gpu_memory >= 8:
        batch_ratio = 4
    elif gpu_memory >= 6:
        batch_ratio = 2
    else:
        batch_ratio = 1
    # logger.info(
    #         f'GPU Memory: {gpu_memory} GB, Batch Ratio: {batch_ratio}. '
    # )

    # 检测torch的版本号
    import torch
    from packaging import version
    device_type = os.getenv("MINERU_LMDEPLOY_DEVICE", "")
    if (
            version.parse(torch.__version__) >= version.parse("2.8.0")
            or str(device).startswith('mps')
            or device_type.lower() in ["corex"]
    ):
        enable_ocr_det_batch = False
    else:
        enable_ocr_det_batch = True
    return batch_ratio, enable_ocr_det_batch


def batch_image_analyze(batch_model, images_with_extra_info: List[Tuple[Image.Image, bool, str]]):
    results = batch_model(images_with_extra_info)
    clean_memory(get_device())
    return results