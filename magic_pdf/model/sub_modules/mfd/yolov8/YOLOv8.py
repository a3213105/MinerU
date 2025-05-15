from tqdm import tqdm
from ultralytics import YOLO
import os
import time
import torch
from ...ov_operator_async import YoloProcessor

class YOLOv8MFDModel(object):
    def __init__(self, weight, enable_ov, enable_bf16, device="cpu"):
        self.mfd_model = YOLO(weight, task="detect")
        self.device = device
        self.enable_ov = enable_ov
        self.enable_bf16 = enable_bf16
        file_name = os.path.basename(weight)
        file_name_without_extension = os.path.splitext(file_name)[0]
        self.ov_file_name = f"{weight}/{file_name_without_extension}.xml".replace(".pt", "_openvino_model")
        # self.ov_file_name = f"{weight}".replace(".pt", ".onnx")
        if self.enable_ov:
            if not os.path.isfile(self.ov_file_name) :
                    path = self.mfd_model.export(format="openvino", dynamic=True, simplify=False, device="CPU")  
                    print(f"### export YOLO from {weight} to {path}, ov_file={self.ov_file_name}")
            self.ov_yolo = YoloProcessor(self.ov_file_name)
            self.ov_yolo.setup_model(stream_num = 1, bf16=self.enable_bf16)
            args={'task': 'detect', 'imgsz': 1888, 'conf': 0.25, 'iou': 0.45, 'batch': 1, 'mode': 'predict',
                  'verbose': False, 'single_cls': False, 'save': False, 'rect': True, 'device': 'cpu'}
            self.mfd_model.predictor = (self.mfd_model._smart_load("predictor"))(overrides=args)
            self.mfd_model.predictor.setup_model(model=self.mfd_model.model, verbose=False)
            def infer(*args):
                result = self.ov_yolo(args)
                return torch.from_numpy(result[0])
            self.mfd_model.predictor.inference = infer
            # self.mfd_model.predictor.model.pt = False
            print(f"### load MFD YOLOV8 Openvino model from {self.ov_file_name}, bf16={self.enable_bf16}")                
        else :
            self.ov_yolo = None

    def predict(self, image):
        # print("### YOLOv8 predict imgsz={imgsz.shape}")
        mfd_res = self.mfd_model.predict(
                image, imgsz=1888, conf=0.25, iou=0.45, verbose=False, device=self.device
            )[0]
        return mfd_res

    def batch_predict(self, images: list, batch_size: int) -> list:
        images_mfd_res = []
        if self.ov_yolo is not None :
            for index in tqdm(range(0, len(images), batch_size), desc="MFD_OV Predict"):
                mfd_res = [
                    image_res.cpu()
                    for image_res in self.mfd_model.predict(
                        images[index : index + batch_size],
                        imgsz=1888,
                        conf=0.25,
                        iou=0.45,
                        verbose=False,
                        device=self.device,
                    )
                ]
                for image_res in mfd_res:
                    images_mfd_res.append(image_res)
                    # print(f"### image_res.boxes={image_res.boxes.shape[0]}")
        # elif self.enable_bf16 :
        #     for index in tqdm(range(0, len(images), batch_size), desc="MFD_BF16 Predict"):
        #         with torch.no_grad(), torch.amp.autocast('cpu'):
        #             image_res = self.mfd_model.predict(
        #                 images[index : index + batch_size],
        #                 imgsz=1888,
        #                 conf=0.25,
        #                 iou=0.45,
        #                 verbose=False,
        #                 device=self.device,
        #             )
        #             for image_res in image_res:
        #                 images_mfd_res.append(image_res)
        #                 print(f"### image_res.boxes={image_res.boxes.shape}")
        else :
            for index in tqdm(range(0, len(images), batch_size), desc="MFD Predict"):
                mfd_res = [
                    image_res.cpu()
                    for image_res in self.mfd_model.predict(
                        images[index : index + batch_size],
                        imgsz=1888,
                        conf=0.25,
                        iou=0.45,
                        verbose=False,
                        device=self.device,
                    )
                ]
                for image_res in mfd_res:
                    images_mfd_res.append(image_res)
                    # print(f"### image_res.boxes={image_res.boxes.shape[0]}")
        # print(f"### YOLOv8 batch_predict done, images_mfd_res={len(images_mfd_res)}")
        return images_mfd_res
