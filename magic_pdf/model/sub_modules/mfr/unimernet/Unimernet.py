import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import os
import warnings

class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
            return image


class UnimernetModel(object):
    def __init__(self, weight_dir, cfg_path, enable_ov, enable_bf16, _device_="cpu"):
        self.enable_ov = enable_ov
        self.enable_bf16 = enable_bf16
        self.ov_file_name = f"{weight_dir}.xml"
        if self.enable_ov and os.path.isfile(self.ov_file_name):
            self.ov_unimernet = UnimernetModel(self.ov_file_name)
            self.ov_unimernet.setup_model(stream_num = 4, bf16=True)
        else:
            from .unimernet_hf import UnimernetModel
            self.ov_unimernet = None
            # print(f"### import UnimernetModel from {weight_dir}, self.enable_bf16={self.enable_bf16}")
            if _device_.startswith("mps"):
                self.model = UnimernetModel.from_pretrained(weight_dir, attn_implementation="eager")
            else:
                self.model = UnimernetModel.from_pretrained(weight_dir)
            self.device = _device_
            self.model.to(_device_)
            if not _device_.startswith("cpu"):
                self.model = self.model.to(dtype=torch.float16)
            self.model.eval()


    def predict(self, mfd_res, image):
        formula_list = []
        mf_image_list = []
        for xyxy, conf, cla in zip(
            mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()
        ):
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                "category_id": 13 + int(cla.item()),
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 2),
                "latex": "",
            }
            formula_list.append(new_item)
            bbox_img = image[ymin:ymax, xmin:xmax]
            mf_image_list.append(bbox_img)

        dataset = MathDataset(mf_image_list, transform=self.model.transform)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=0)
        mfr_res = []
        for mf_img in dataloader:
            mf_img = mf_img.to(dtype=self.model.dtype)
            mf_img = mf_img.to(self.device)
            with torch.no_grad():
                output = self.model.generate({"image": mf_img})
            mfr_res.extend(output["fixed_str"])
        for res, latex in zip(formula_list, mfr_res):
            res["latex"] = latex
        return formula_list

    def batch_predict(self, images_mfd_res: list, images: list, batch_size: int = 64) -> list:
        images_formula_list = []
        mf_image_list = []
        backfill_list = []
        image_info = []  # Store (area, original_index, image) tuples

        # Collect images with their original indices
        # print(f"### images_mfd_res={len(images_mfd_res)}")
        t0 = time.time()
        for image_index in range(len(images_mfd_res)):
            mfd_res = images_mfd_res[image_index]
            np_array_image = images[image_index]
            formula_list = []

            for idx, (xyxy, conf, cla) in enumerate(zip(
                    mfd_res.boxes.xyxy, mfd_res.boxes.conf, mfd_res.boxes.cls
            )):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                new_item = {
                    "category_id": 13 + int(cla.item()),
                    "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    "score": round(float(conf.item()), 2),
                    "latex": "",
                }
                formula_list.append(new_item)
                bbox_img = np_array_image[ymin:ymax, xmin:xmax]
                area = (xmax - xmin) * (ymax - ymin)

                curr_idx = len(mf_image_list)
                image_info.append((area, curr_idx, bbox_img))
                mf_image_list.append(bbox_img)

            images_formula_list.append(formula_list)
            backfill_list += formula_list
        t1 = time.time()
        # Stable sort by area
        image_info.sort(key=lambda x: x[0])  # sort by area
        sorted_indices = [x[1] for x in image_info]
        sorted_images = [x[2] for x in image_info]

        # Create mapping for results
        index_mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(sorted_indices)}

        # Create dataset with sorted images
        dataset = MathDataset(sorted_images, transform=self.model.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        # Process batches and store results
        mfr_res = []
        # for mf_img in dataloader:
        t2 = time.time()
        if self.ov_unimernet is not None:
            self.ov_unimernet(sorted_images)
            mfr_res.extend(output["fixed_str"])
        elif self.enable_bf16 :
            with tqdm(total=len(sorted_images), desc="MFR_BF16 Predict") as pbar:
                for index, mf_img in enumerate(dataloader):
                    mf_img = mf_img.to(dtype=self.model.dtype)
                    mf_img = mf_img.to(self.device)
                    with torch.no_grad(), torch.amp.autocast('cpu'):
                        # output = self.model.generate({"image": mf_img})
                        outputs = self.model.generate(mf_img)
                        outputs = outputs.cpu().numpy()
                        output = self.model.parser_result(outputs)

                    # mfr_res.extend(output["fixed_str"])
                    mfr_res.extend(output)

                    # 更新进度条，每次增加batch_size，但要注意最后一个batch可能不足batch_size
                    current_batch_size = min(batch_size, len(sorted_images) - index * batch_size)
                    pbar.update(current_batch_size)
        else :
            with tqdm(total=len(sorted_images), desc="MFR Predict") as pbar:
                for index, mf_img in enumerate(dataloader):
                    mf_img = mf_img.to(dtype=self.model.dtype)
                    mf_img = mf_img.to(self.device)
                    class model_wrapper(torch.nn.Module) :
                        def __init__(self, ipm):
                            super(model_wrapper, self).__init__()
                            self.mm = ipm
                        def forward(self, mf_img):
                            return self.mm.generate(mf_img)
                    mm = model_wrapper(self.model)
                    mm = mm.eval()
                    with torch.no_grad():
                        # output = self.model.generate({"image": mf_img})
                        outputs = mm(mf_img)
                        outputs = outputs.cpu().numpy()
                        output = self.model.parser_result(outputs)
                        # print(f"### UnimernetModel batch_predict: mf_img={mf_img.shape}, output={output}")
                        # with warnings.catch_warnings():
                        #     # warnings.filterwarnings("ignore")
                        #     if not os.path.isfile(self.ov_file_name):
                        #         print(f"save {self.ov_file_name}, mf_img={mf_img.shape}")
                        #         # try:
                        #         onnx_program = torch.onnx.export(mm, mf_img, dynamo=False, report=True, strict=False)
                        #             # print(f"### UnimernetModel batch_predict: onnx_program={onnx_program}")
                        #             # onnx_path = "/tmp/test.onnx"
                        #             # torch.onnx.export(mm, mf_img, onnx_path,)
                        #             # print(f"ONNX model exported to {onnx_path}.")
                        #         #     import openvino as ov
                        #         #     ov_model = ov.convert_model(mm, example_input=mf_img)
                        #         #     ov.save_model(ov_model, self.ov_file_name)
                        #         # except Exception as e:
                        #         #     print(f"### UnimernetModel convert_model failed: {e}, try simple convert_model")
                        #         #     exit(-1)

                    # mfr_res.extend(output["fixed_str"])
                    mfr_res.extend(output)

                    # 更新进度条，每次增加batch_size，但要注意最后一个batch可能不足batch_size
                    current_batch_size = min(batch_size, len(sorted_images) - index * batch_size)
                    pbar.update(current_batch_size)
        t3 = time.time()
        # Restore original order
        unsorted_results = [""] * len(mfr_res)
        for new_idx, latex in enumerate(mfr_res):
            original_idx = index_mapping[new_idx]
            unsorted_results[original_idx] = latex
        # Fill results back
        for res, latex in zip(backfill_list, unsorted_results):
            res["latex"] = latex
        t4 = time.time()
        # print(f"### UnimernetModel batch_predict: images_mfd_res={len(images_mfd_res)}, images={len(images)}, ",
        #       f"batch_size={batch_size}, sorted_images={len(sorted_images)}, ",
        #       f"{t1-t0:.6f} {t2-t1:.6f} {t3-t2:.6f} {t4-t3:.6f} ",)
        return images_formula_list
