import cv2
import copy
import numpy as np
import math
import time
import torch
from ...pytorchocr.base_ocr_v20 import BaseOCRV20
from . import pytorchocr_utility as utility
from ...pytorchocr.postprocess import build_post_process

using_ov = True
try :
    from .....ov_operator_async import PaddleTextClsProcessor
except ImportError as e:
    using_ov = False
    print(f"### import ov_operator_async failed, {e}")

class TextClassifier(BaseOCRV20):
    def __init__(self, args, **kwargs):
        self.device = args.device
        self.cls_image_shape = [int(v) for v in args.cls_image_shape.split(",")]
        self.cls_batch_num = args.cls_batch_num
        self.cls_thresh = args.cls_thresh
        postprocess_params = {
            'name': 'ClsPostProcess',
            "label_list": args.label_list,
        }
        self.postprocess_op = build_post_process(postprocess_params)

        self.weights_path = args.cls_model_path
        self.yaml_path = args.cls_yaml_path
        network_config = utility.get_arch_config(self.weights_path)
        super(TextClassifier, self).__init__(network_config, **kwargs)

        self.cls_image_shape = [int(v) for v in args.cls_image_shape.split(",")]

        self.limited_max_width = args.limited_max_width
        self.limited_min_width = args.limited_min_width

        try :
            self.enable_ov = args.enable_ov and using_ov
        except AttributeError:
            self.enable_ov = True
            
        try :
            self.enable_bf16 = False #args.enable_bf16_rec
        except AttributeError:
            self.enable_bf16 = False
        self.ov_file_name = f"{self.weights_path}.xml"
        self.ov_cls = None
        if self.enable_ov:
            if not os.path.isfile(self.ov_file_name):
                self.load_pytorch_weights(self.weights_path)
                self.net.eval()
                self.net.to(self.device)
                import openvino as ov
                try :
                    ov_model = ov.convert_model(self.net, example_input=torch.randn(1, 3, 960, 960))
                    ov.save_model(ov_model, self.ov_file_name)
                    print(f"export ov model to {self.ov_file_name} ")
                except Exception as e:
                    print(f"### convert_model failed: {e}, try simple convert_model")
            if os.path.isfile(self.ov_file_name):
                self.ov_cls = PaddleTextDetector(self.ov_file_name)
                self.ov_cls.setup_model(stream_num = 1, bf16=self.enable_bf16,) 
                                        # shape_dynamic=[1, self.rec_image_shape[1], -1, self.rec_image_shape[0]])
                print(f"### load OCR-Cls_ov model {self.ov_file_name}, enable_bf16={self.enable_bf16}, ",
                    f"det_algorithm={self.det_algorithm}")
        else :
            self.load_pytorch_weights(self.weights_path)
            self.net.eval()
            self.net.to(self.device)
        # print(f"### TextClassifier init enable_ov={self.enable_ov}, enable_bf16={self.enable_bf16}")

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        imgW = max(min(imgW, self.limited_max_width), self.limited_min_width)
        ratio_imgH = math.ceil(imgH * ratio)
        ratio_imgH = max(ratio_imgH, self.limited_min_width)
        if ratio_imgH > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        cls_res = [['', 0.0]] * img_num
        batch_num = self.cls_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            starttime = time.time()

            inp = torch.from_numpy(norm_img_batch)
            inp = inp.to(self.device)
            if self.ov_cls is None:
                if self.enable_bf16:
                    with torch.no_grad(), torch.amp.autocast('cpu'):
                        prob_out = self.net(inp)
                else:
                    with torch.no_grad():
                        prob_out = self.net(inp)
                if self.enable_ov and not os.path.isfile(self.ov_file_name):
                    import openvino as ov
                    ov_model = ov.convert_model(self.net, example_input=inp)
                    ov.save_model(ov_model, self.ov_file_name)
                    print(f"export ov model to {self.ov_file_name} with example_input={inp.shape}")
            else:
                prob_out = self.ov_cls(inp)

            prob_out = prob_out.cpu().float().numpy()

            cls_result = self.postprocess_op(prob_out)
            elapse += time.time() - starttime
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if '180' in label and score > self.cls_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1)
        return img_list, cls_res, elapse
