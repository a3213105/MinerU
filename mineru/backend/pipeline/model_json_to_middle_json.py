# Copyright (c) Opendatalab. All rights reserved.
import os
import time

from loguru import logger
from tqdm import tqdm

from mineru.backend.utils import cross_page_table_merge
from mineru.utils.config_reader import get_device, get_llm_aided_config, get_formula_enable
from mineru.backend.pipeline.model_init import AtomModelSingleton
from mineru.backend.pipeline.para_split import para_split
from mineru.utils.block_pre_proc import prepare_block_bboxes, process_groups
from mineru.utils.block_sort import sort_blocks_by_bbox
from mineru.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType, ImageType
from mineru.utils.pdf_image_tools import load_image_from_pdf
from mineru.utils.llm_aided import llm_aided_title
from mineru.utils.model_utils import clean_memory
from mineru.backend.pipeline.pipeline_magic_model import MagicModel
from mineru.utils.ocr_utils import OcrConfidence
from mineru.utils.span_block_fix import fill_spans_in_blocks, fix_discarded_block, fix_block_spans
from mineru.utils.span_pre_proc import remove_outside_spans, remove_overlaps_low_confidence_spans, \
    remove_overlaps_min_spans, txt_spans_extract
from mineru.version import __version__
from mineru.utils.hash_utils import bytes_md5


# def page_model_info_to_page_info(page_model_info, image_dict, page, image_writer, page_index, ocr_enable=False, formula_enabled=True, enable_ov, infer_type):
def page_model_info_to_page_info(page_model_info, image_dict, page, image_writer, page_index, ocr_enable, formula_enabled, batch_model):
    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    # page_img_md5 = str_md5(image_dict["img_base64"])
    page_img_md5 = bytes_md5(page_pil_img.tobytes())
    page_w, page_h = map(int, page.get_size())
    magic_model = MagicModel(page_model_info, scale)

    """从magic_model对象中获取后面会用到的区块信息"""
    discarded_blocks = magic_model.get_discarded()
    text_blocks = magic_model.get_text_blocks()
    title_blocks = magic_model.get_title_blocks()
    inline_equations, interline_equations, interline_equation_blocks = magic_model.get_equations()

    img_groups = magic_model.get_imgs()
    table_groups = magic_model.get_tables()

    """对image和table的区块分组"""
    img_body_blocks, img_caption_blocks, img_footnote_blocks, maybe_text_image_blocks = process_groups(
        img_groups, 'image_body', 'image_caption_list', 'image_footnote_list'
    )

    table_body_blocks, table_caption_blocks, table_footnote_blocks, _ = process_groups(
        table_groups, 'table_body', 'table_caption_list', 'table_footnote_list'
    )

    """获取所有的spans信息"""
    spans = magic_model.get_all_spans()

    """某些图可能是文本块，通过简单的规则判断一下"""
    if len(maybe_text_image_blocks) > 0:
        for block in maybe_text_image_blocks:
            should_add_to_text_blocks = False

            if ocr_enable:
                # 找到与当前block重叠的text spans
                span_in_block_list = [
                    span for span in spans
                    if span['type'] == 'text' and
                       calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block['bbox']) > 0.7
                ]

                if len(span_in_block_list) > 0:
                    # 计算spans总面积
                    spans_area = sum(
                        (span['bbox'][2] - span['bbox'][0]) * (span['bbox'][3] - span['bbox'][1])
                        for span in span_in_block_list
                    )

                    # 计算block面积
                    block_area = (block['bbox'][2] - block['bbox'][0]) * (block['bbox'][3] - block['bbox'][1])

                    # 判断是否符合文本图条件
                    if block_area > 0 and spans_area / block_area > 0.25:
                        should_add_to_text_blocks = True

            # 根据条件决定添加到哪个列表
            if should_add_to_text_blocks:
                block.pop('group_id', None)  # 移除group_id
                text_blocks.append(block)
            else:
                img_body_blocks.append(block)


    """将所有区块的bbox整理到一起"""
    if formula_enabled:
        interline_equation_blocks = []

    if len(interline_equation_blocks) > 0:
        for block in interline_equation_blocks:
            spans.append({
                "type": ContentType.INTERLINE_EQUATION,
                'score': block['score'],
                "bbox": block['bbox'],
                "content": "",
            })
        all_bboxes, all_discarded_blocks, footnote_blocks = prepare_block_bboxes(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equation_blocks,
            page_w,
            page_h,
        )
    else:
        all_bboxes, all_discarded_blocks, footnote_blocks = prepare_block_bboxes(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equations,
            page_w,
            page_h,
        )

    """在删除重复span之前，应该通过image_body和table_body的block过滤一下image和table的span"""
    """顺便删除大水印并保留abandon的span"""
    spans = remove_outside_spans(spans, all_bboxes, all_discarded_blocks)

    """删除重叠spans中置信度较低的那些"""
    spans, dropped_spans_by_confidence = remove_overlaps_low_confidence_spans(spans)
    """删除重叠spans中较小的那些"""
    spans, dropped_spans_by_span_overlap = remove_overlaps_min_spans(spans)

    """根据parse_mode，构造spans，主要是文本类的字符填充"""
    if ocr_enable:
        pass
    else:
        """使用新版本的混合ocr方案."""
        spans = txt_spans_extract(page, spans, page_pil_img, scale, all_bboxes, all_discarded_blocks)

    """先处理不需要排版的discarded_blocks"""
    discarded_block_with_spans, spans = fill_spans_in_blocks(
        all_discarded_blocks, spans, 0.4
    )
    fix_discarded_blocks = fix_discarded_block(discarded_block_with_spans)

    """如果当前页面没有有效的bbox则跳过"""
    if len(all_bboxes) == 0 and len(fix_discarded_blocks) == 0:
        return None

    """对image/table/interline_equation截图"""
    for span in spans:
        if span['type'] in [ContentType.IMAGE, ContentType.TABLE, ContentType.INTERLINE_EQUATION]:
            span = cut_image_and_table(
                span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale
            )

    """span填充进block"""
    print(f"page {page_index} all_bboxes: {len(all_bboxes)}, spans: {len(spans)}")  # --- IGNORE ---
    block_with_spans, spans = fill_spans_in_blocks(all_bboxes, spans, 0.5)
    print(f"page {page_index} block_with_spans: {len(block_with_spans)}, spans not in blocks: {len(spans)}")  # --- IGNORE ---

    """对block进行fix操作"""
    fix_blocks = fix_block_spans(block_with_spans)

    """对block进行排序"""
    sorted_blocks = sort_blocks_by_bbox(fix_blocks, page_w, page_h, footnote_blocks, batch_model)

    """构造page_info"""
    page_info = make_page_info_dict(sorted_blocks, page_index, page_w, page_h, fix_discarded_blocks)

    return page_info


def result_to_middle_json(model_list, images_list, pdf_doc, image_writer, batch_model, lang=None,
                          ocr_enable=False, formula_enabled=True, tqdm_enable: bool = False,
                          pdf_bytes=None):
    middle_json = {"pdf_info": [], "_backend":"pipeline", "_version_name": __version__}
    formula_enabled = get_formula_enable(formula_enabled)
    tpdm_desc = f"Processing pages with OV_{batch_model.OCR_rec_infer_type}" if batch_model.enable_ov else "Processing pages"
    for page_index, page_model_info in tqdm(enumerate(model_list), total=len(model_list), desc=tpdm_desc, disable=not tqdm_enable):
        page = pdf_doc[page_index]
        if images_list is None:
            if pdf_bytes is None:
                raise ValueError("pdf_bytes is required when images_list is None")
            image_dict = load_image_from_pdf(pdf_bytes, pdf_doc, image_type=ImageType.PIL,
                                             start_page_id=page_index, end_page_id=page_index,)[0]
        else:
            image_dict = images_list[page_index]
        page_info = page_model_info_to_page_info(page_model_info, image_dict, page, image_writer, page_index, ocr_enable=ocr_enable, formula_enabled=formula_enabled, batch_model=batch_model)
        if page_info is None:
            page_w, page_h = map(int, page.get_size())
            page_info = make_page_info_dict([], page_index, page_w, page_h, [])
        middle_json["pdf_info"].append(page_info)
        del image_dict
    if images_list is not None:
        images_list.clear()
    """后置ocr处理"""
    ocr_model = batch_model.model.get_ocr_model(det_db_box_thresh=0.3, lang=lang,)

    def apply_ocr_to_chunk(span_list, image_list):
        if not image_list:
            return
        ocr_res_list = ocr_model.ocr(image_list, det=False, tqdm_enable=tqdm_enable)[0]
        assert len(ocr_res_list) == len(
            span_list), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(span_list)}'
        for span, (ocr_text, ocr_score) in zip(span_list, ocr_res_list):
            if ocr_score > OcrConfidence.min_confidence:
                span['content'] = ocr_text
                span['score'] = float(f"{ocr_score:.3f}")
            else:
                span['content'] = ''
                span['score'] = 0.0

    if batch_model.enable_cache:
        need_ocr_list = []
        img_crop_list = []
        for page_info in middle_json["pdf_info"]:
            for block in page_info['preproc_blocks']:
                if block['type'] in ['table', 'image']:
                    for sub_block in block['blocks']:
                        if sub_block['type'] in ['image_caption', 'image_footnote', 'table_caption', 'table_footnote']:
                            for line in sub_block['lines']:
                                for span in line['spans']:
                                    if 'np_img' in span:
                                        need_ocr_list.append(span)
                                        img_crop_list.append(span['np_img'])
                                        span.pop('np_img')
                elif block['type'] in ['text', 'title']:
                    for line in block['lines']:
                        for span in line['spans']:
                            if 'np_img' in span:
                                need_ocr_list.append(span)
                                img_crop_list.append(span['np_img'])
                                span.pop('np_img')
            for block in page_info['discarded_blocks']:
                for line in block['lines']:
                    for span in line['spans']:
                        if 'np_img' in span:
                            need_ocr_list.append(span)
                            img_crop_list.append(span['np_img'])
                            span.pop('np_img')
        apply_ocr_to_chunk(need_ocr_list, img_crop_list)
    else:
        chunk_size = int(os.environ.get('MINERU_OCR_REC_CHUNK_SIZE', 64))
        chunk_spans = []
        chunk_images = []
        for page_info in middle_json["pdf_info"]:
            for block in page_info['preproc_blocks']:
                candidate_blocks = []
                if block['type'] in ['table', 'image']:
                    candidate_blocks.extend(
                        sub_block for sub_block in block['blocks']
                        if sub_block['type'] in ['image_caption', 'image_footnote', 'table_caption', 'table_footnote']
                    )
                elif block['type'] in ['text', 'title']:
                    candidate_blocks.append(block)

                for text_block in candidate_blocks:
                    for line in text_block['lines']:
                        for span in line['spans']:
                            if 'np_img' in span:
                                chunk_spans.append(span)
                                chunk_images.append(span.pop('np_img'))
                                if len(chunk_images) >= chunk_size:
                                    apply_ocr_to_chunk(chunk_spans, chunk_images)
                                    chunk_spans.clear()
                                    chunk_images.clear()

            for block in page_info['discarded_blocks']:
                for line in block['lines']:
                    for span in line['spans']:
                        if 'np_img' in span:
                            chunk_spans.append(span)
                            chunk_images.append(span.pop('np_img'))
                            if len(chunk_images) >= chunk_size:
                                apply_ocr_to_chunk(chunk_spans, chunk_images)
                                chunk_spans.clear()
                                chunk_images.clear()

        if chunk_images:
            apply_ocr_to_chunk(chunk_spans, chunk_images)

    """分段"""
    para_split(middle_json["pdf_info"])

    """表格跨页合并"""
    cross_page_table_merge(middle_json["pdf_info"])

    """llm优化"""
    llm_aided_config = get_llm_aided_config()

    if llm_aided_config is not None:
        """标题优化"""
        title_aided_config = llm_aided_config.get('title_aided', None)
        if title_aided_config is not None:
            if title_aided_config.get('enable', False):
                llm_aided_title_start_time = time.time()
                llm_aided_title(middle_json["pdf_info"], title_aided_config)
                logger.info(f'llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}')

    """清理内存"""
    pdf_doc.close()
    if os.getenv('MINERU_DONOT_CLEAN_MEM') is None and len(model_list) >= 10:
        clean_memory(get_device())

    return middle_json


def make_page_info_dict(blocks, page_id, page_w, page_h, discarded_blocks):
    return_dict = {
        'preproc_blocks': blocks,
        'page_idx': page_id,
        'page_size': [page_w, page_h],
        'discarded_blocks': discarded_blocks,
    }
    return return_dict