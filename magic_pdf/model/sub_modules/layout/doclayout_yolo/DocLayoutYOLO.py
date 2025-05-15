from doclayout_yolo import YOLOv10
from tqdm import tqdm
from ...ov_operator_async import YoloProcessor
import os
import torch

class DocLayoutYOLOModel(object):
    def __init__(self, weight, enable_ov, enable_bf16, device):
        self.model = YOLOv10(weight)
        self.device = device
        self.enable_ov = enable_ov
        self.enable_bf16 = enable_bf16
        file_name = os.path.basename(weight)
        file_name_without_extension = os.path.splitext(file_name)[0]
        self.ov_file_name = f"{weight}/{file_name_without_extension}.xml".replace(".pt", "_openvino_model")
        if self.enable_ov:
            if not os.path.isfile(self.ov_file_name) :
                    path = self.model.export(format="openvino", dynamic=True)  
                    print(f"### export YOLOv10 from {weight} to {path}, ov_file={self.ov_file_name}")
            self.ov_yolo = YoloProcessor(self.ov_file_name)
            self.ov_yolo.setup_model(stream_num = 1, bf16=self.enable_bf16)
            args={'task': 'detect', 'imgsz': 1280, 'conf': 0.10, 'iou': 0.45, 'batch': 1, 'mode': 'predict',
                  'verbose': False, 'single_cls': False, 'save': False, 'rect': True, 'device': 'cpu'}
            self.model.predictor = (self.model._smart_load("predictor"))(overrides=args)
            self.model.predictor.setup_model(model=self.model.model, verbose=False)
            def infer(*args):
                result = self.ov_yolo(args)
                return torch.from_numpy(result[0])
            self.model.predictor.inference = infer
            # self.model.predictor.model.pt = False
            print(f"### load DocLayoutYOLO Openvino model from {self.ov_file_name}, bf16={self.enable_bf16}")                
        else :
            self.ov_yolo = None

    def predict(self, image):
        layout_res = []
        doclayout_yolo_res = self.model.predict(
            image,
            imgsz=1280,
            conf=0.10,
            iou=0.45,
            verbose=False, device=self.device
        )[0]
        for xyxy, conf, cla in zip(
            doclayout_yolo_res.boxes.xyxy.cpu(),
            doclayout_yolo_res.boxes.conf.cpu(),
            doclayout_yolo_res.boxes.cls.cpu(),
        ):
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                "category_id": int(cla.item()),
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 3),
            }
            layout_res.append(new_item)
        return layout_res

    def batch_predict(self, images: list, batch_size: int) -> list:
        images_layout_res = []
        # for index in range(0, len(images), batch_size):
        if self.ov_yolo is None :
            if self.enable_bf16:
                for index in tqdm(range(0, len(images), batch_size), desc="Layout_bf16 Predict"):
                    with torch.no_grad(), torch.amp.autocast('cpu'):
                        doclayout_yolo_res = [
                            image_res.cpu()
                            for image_res in self.model.predict(
                                images[index : index + batch_size],
                                imgsz=1280,
                                conf=0.10,
                                iou=0.45,
                                verbose=False,
                                device=self.device,
                            )
                        ]
                        for image_res in doclayout_yolo_res:
                            layout_res = []
                            for xyxy, conf, cla in zip(
                                image_res.boxes.xyxy,
                                image_res.boxes.conf,
                                image_res.boxes.cls,
                            ):
                                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                                new_item = {
                                    "category_id": int(cla.item()),
                                    "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                                    "score": round(float(conf.item()), 3),
                                }
                                layout_res.append(new_item)
                            images_layout_res.append(layout_res)
            else :
                for index in tqdm(range(0, len(images), batch_size), desc="Layout Predict"):
                    doclayout_yolo_res = [
                        image_res.cpu()
                        for image_res in self.model.predict(
                            images[index : index + batch_size],
                            imgsz=1280,
                            conf=0.10,
                            iou=0.45,
                            verbose=False,
                            device=self.device,
                        )
                    ]
                    for image_res in doclayout_yolo_res:
                        layout_res = []
                        for xyxy, conf, cla in zip(
                            image_res.boxes.xyxy,
                            image_res.boxes.conf,
                            image_res.boxes.cls,
                        ):
                            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                            new_item = {
                                "category_id": int(cla.item()),
                                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                                "score": round(float(conf.item()), 3),
                            }
                            layout_res.append(new_item)
                        images_layout_res.append(layout_res)
        else :
            for index in tqdm(range(0, len(images), batch_size), desc="Layout_OV Predict"):
                doclayout_yolo_res = [
                    image_res.cpu()
                    for image_res in self.model.predict(
                        images[index : index + batch_size],
                        imgsz=1280,
                        conf=0.10,
                        iou=0.45,
                        verbose=False,
                        device=self.device,
                    )
                ]
                for image_res in doclayout_yolo_res:
                    layout_res = []
                    for xyxy, conf, cla in zip(
                        image_res.boxes.xyxy,
                        image_res.boxes.conf,
                        image_res.boxes.cls,
                    ):
                        xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                        new_item = {
                            "category_id": int(cla.item()),
                            "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                            "score": round(float(conf.item()), 3),
                        }
                        layout_res.append(new_item)
                    images_layout_res.append(layout_res)
        return images_layout_res
