from typing import *
from numpy import ndarray
from torch import Tensor

import os
import json
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as tF
from kiui.cam import orbit_camera, undo_orbit_camera

from src.data.utils.chunk_dataset import ChunkedDataset
from src.options import Options
from src.utils import normalize_normals, unproject_depth


class GObjaverseParquetDataset(ChunkedDataset):
    def __init__(self, opt: Options, training: bool = True, *args, **kwargs):
        self.opt = opt
        self.training = training

        # Default camera intrinsics
        self.fxfycxcy = torch.tensor([opt.fxfy, opt.fxfy, 0.5, 0.5], dtype=torch.float32)  # (4,)

        if opt.prompt_embed_dir is not None:
            try:
                self.negative_prompt_embed = torch.from_numpy(np.load(f"{opt.prompt_embed_dir}/null.npy")).float()
            except FileNotFoundError:
                self.negative_prompt_embed = None
            try:
                self.negative_pooled_prompt_embed = torch.from_numpy(np.load(f"{opt.prompt_embed_dir}/null_pooled.npy")).float()
            except FileNotFoundError:
                self.negative_pooled_prompt_embed = None
            try:
                self.negative_prompt_attention_mask = torch.from_numpy(np.load(f"{opt.prompt_embed_dir}/null_attention_mask.npy")).float()
            except FileNotFoundError:
                self.negative_prompt_attention_mask = None

            if "xl" in opt.pretrained_model_name_or_path:  # SDXL: zero out negative prompt embedding
                if self.negative_prompt_embed is not None and self.negative_pooled_prompt_embed is not None:
                    self.negative_prompt_embed = torch.zeros_like(self.negative_prompt_embed)
                    self.negative_pooled_prompt_embed = torch.zeros_like(self.negative_pooled_prompt_embed)

        # Backup from local disk for error data loading
        with open(opt.backup_json_path, "r") as f:
            self.backup_ids = json.load(f)

        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.opt.dataset_size

    def get_trainable_data_from_raw_data(self, raw_data_list) -> Dict[str, Tensor]:  # only `sample["__key__"]` is in str type
        assert len(raw_data_list) == 1
        sample: Dict[str, bytes] = raw_data_list[0]

        V, V_in = self.opt.num_views, self.opt.num_input_views
        assert V >= V_in

        if self.opt.load_even_views or not self.training:
            _pick_func = self._pick_even_view_indices
        else:
            _pick_func = self._pick_random_view_indices

        # Randomly sample `V_in` views (some objects may not appear in the dataset)
        random_idxs = _pick_func(V_in)
        _num_tries = 0
        while not self._check_views_exist(sample, random_idxs):
            random_idxs = _pick_func(V_in)
            _num_tries += 1
            if _num_tries > 100:  # TODO: make `100` configurable
                raise ValueError(f"Cannot find 4 views in {sample['__key__']}")

        except_idxs = random_idxs + [24, 39]  # filter duplicated views; hard-coded for GObjaverse
        if self.opt.exclude_topdown_views:
            except_idxs += [25, 26]

        # Randomly sample `V` views (some views may not appear in the dataset)
        for i in np.random.permutation(40):  # `40` is hard-coded for GObjaverse
            if len(random_idxs) >= V:
                break
            if f"{i:05d}.png" in sample and i not in except_idxs:
                try:
                    _ = np.frombuffer(sample[f"{i:05d}.png"], np.uint8)
                    assert sample[f"{i:05d}.json"] is not None
                    random_idxs.append(i)
                except:  # TypeError: a bytes-like object is required, not 'NoneType'; KeyError: '00001.json'
                    pass
        # Randomly repeat views if not enough views
        while len(random_idxs) < V:
            random_idxs.append(np.random.choice(random_idxs))

        return_dict = defaultdict(list)
        init_azi = None
        for vid in random_idxs:
            return_dict["fxfycxcy"].append(self.fxfycxcy)  # (V, 4); fixed intrinsics for GObjaverse

            image = self._load_png(sample[f"{vid:05d}.png"])  # (4, 512, 512)
            mask = image[3:4]  # (1, 512, 512)
            image = image[:3] * mask + (1. - mask)  # (3, 512, 512), to white bg
            return_dict["image"].append(image)  # (V, 3, H, W)
            return_dict["mask"].append(mask)  # (V, 1, H, W)

            if self.opt.load_canny:
                gray = cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
                canny = cv2.Canny((gray * 255.).astype(np.uint8), 100., 200.)
                canny = torch.from_numpy(canny).unsqueeze(0).float().repeat(3, 1, 1) / 255.  # (3, 512, 512) in [0, 1]
                canny = -canny + 1.  # 0->1, 1->0, i.e., white bg
                return_dict["canny"].append(canny)  # (V, 3, H, W)

            if not USE_BACKUP:
                c2w = self._load_camera_from_json(sample[f"{vid:05d}.json"])
            else:
                c2w = self._load_camera_from_json(f"{uid}.{vid:05d}")
            # Blender world + OpenCV cam -> OpenGL world & cam; https://kit.kiui.moe/camera
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1  # invert up and forward direction
            return_dict["original_C2W"].append(torch.from_numpy(c2w).float())  # (V, 4, 4); for normal normalization only

            # Relative azimuth w.r.t. the first view
            ele, azi, dis = undo_orbit_camera(c2w)  # elevation: [-90, 90] from +y(-90) to -y(90)
            if init_azi is None:
                init_azi = azi
            azi = (azi - init_azi) % 360.  # azimuth: [0, 360] from +z(0) to +x(90)
            # To avoid numerical errors for elevation +/- 90 (GObjaverse index 25 (up) & 26 (down))
            ele_sign = ele >= 0
            ele = abs(ele) - 1e-8
            ele = ele * (1. if ele_sign else -1.)

            new_c2w = torch.from_numpy(orbit_camera(ele, azi, dis)).float()
            return_dict["C2W"].append(new_c2w)  # (V, 4, 4)
            return_dict["cam_pose"].append(torch.tensor(
                [np.deg2rad(ele), np.deg2rad(azi), dis], dtype=torch.float32))  # (V, 3)

            # Albedo
            if self.opt.load_albedo:
                albedo = self._load_png(sample[f"{vid:05d}_albedo.png"])  # (3, 512, 512)
                albedo = albedo * mask + (1. - mask)  # (3, 512, 512), to white bg
                return_dict["albedo"].append(albedo)  # (V, 3, H, W)
            # Normal & Depth
            if self.opt.load_normal or self.opt.load_coord:
                nd = self._load_png(sample[f"{vid:05d}_nd.png"], uint16=True)  # (4, 512, 512)
                if self.opt.load_normal:
                    normal = nd[:3] * 2. - 1.  # (3, 512, 512) in [-1, 1]
                    normal[0, ...] *= -1  # to OpenGL world convention
                    return_dict["normal"].append(normal)  # (V, 3, H, W)
                if self.opt.load_coord or self.opt.load_depth:
                    depth = nd[3] * 5.  # (512, 512); NOTE: depth is scaled by 1/5 in my data preprocessing
                    return_dict["depth"].append(depth)  # (V, H, W)
            # Metal & Roughness
            if self.opt.load_mr:
                mr = self._load_png(sample[f"{vid:05d}_mr.png"])  # (3, 512, 512); (metallic, roughness, padding)
                mr = mr * mask + (1. - mask)  # (3, 512, 512), to white bg
                return_dict["mr"].append(mr)  # (V, 3, H, W)

        for key in return_dict.keys():
            return_dict[key] = torch.stack(return_dict[key], dim=0)

        if self.opt.load_normal:
            # Normalize normals by the first view and transform the first view to a fixed azimuth (i.e., 0)
            # Ensure `normals` and `original_C2W` are in the same camera convention
            normals = normalize_normals(return_dict["normal"].unsqueeze(0), return_dict["original_C2W"].unsqueeze(0), i=0).squeeze(0)
            normals = torch.einsum("brc,bvchw->bvrhw", return_dict["C2W"][0, :3, :3].unsqueeze(0), normals.unsqueeze(0)).squeeze(0)
            normals = normals * 0.5 + 0.5  # [0, 1]
            normals = normals * return_dict["mask"] + (1. - return_dict["mask"])  # (V, 3, 512, 512), to white bg
            return_dict["normal"] = normals
            return_dict.pop("original_C2W")  # original C2W is only used for normal normalization

        # OpenGL to COLMAP camera for Gaussian renderer
        return_dict["C2W"][:, :3, 1:3] *= -1

        # Whether scale the object w.r.t. the first view to a fixed size
        if self.opt.norm_camera:
            scale = self.opt.norm_radius / (torch.norm(return_dict["C2W"][0, :3, 3], dim=-1))
        else:
            scale = 1.
        return_dict["C2W"][:, :3, 3] *= scale
        return_dict["cam_pose"][:, 2] *= scale

        if self.opt.load_coord:
            # Unproject depth map to 3D world coordinate
            coords = unproject_depth(return_dict["depth"].unsqueeze(0) * scale,
                return_dict["C2W"].unsqueeze(0), return_dict["fxfycxcy"].unsqueeze(0)).squeeze(0)
            coords = coords * 0.5 + 0.5  # [0, 1]
            coords = coords * return_dict["mask"] + (1. - return_dict["mask"])  # (V, 3, 512, 512), to white bg
            return_dict["coord"] = coords
            if not self.opt.load_depth:
                return_dict.pop("depth")

        if self.opt.load_depth:
            depths = return_dict["depth"].unsqueeze(1) * return_dict["mask"]  # (V, 1, 512, 512), to black bg
            assert depths.min() == 0.
            if self.opt.normalize_depth:
                H, W = depths.shape[-2:]
                depths = depths.reshape(V, -1)
                depths_max = depths.max(dim=-1, keepdim=True).values
                depths = depths / depths_max  # [0, 1]
                depths = depths.reshape(V, 1, H, W)
            depths = -depths + 1.  # 0->1, 1->0, i.e., white bg
            return_dict["depth"] = depths.repeat(1, 3, 1, 1)

        # Resize to the input resolution
        for key in ["image", "mask", "albedo", "normal", "coord", "depth", "mr", "canny"]:
            if key in return_dict.keys():
                return_dict[key] = tF.interpolate(
                    return_dict[key], size=(self.opt.input_res, self.opt.input_res),
                    mode="bilinear", align_corners=False, antialias=True
                )

        # Handle anti-aliased normal, coord and depth (GObjaverse renders anti-aliased normal & depth)
        if self.opt.load_normal:
            return_dict["normal"] = return_dict["normal"] * return_dict["mask"] + (1. - return_dict["mask"])
        if self.opt.load_coord:
            return_dict["coord"] = return_dict["coord"] * return_dict["mask"] + (1. - return_dict["mask"])
        if self.opt.load_depth:
            return_dict["depth"] = return_dict["depth"] * return_dict["mask"] + (1. - return_dict["mask"])

        # Load precomputed caption embeddings
        if self.opt.prompt_embed_dir is not None:
            uid = sample["uid"].decode("utf-8").split("/")[-1].split(".")[0]
            return_dict["prompt_embed"] = torch.from_numpy(np.load(f"{self.opt.prompt_embed_dir}/{uid}.npy"))
            if "xl" in self.opt.pretrained_model_name_or_path or "3" in self.opt.pretrained_model_name_or_path:  # SDXL or SD3
                return_dict["pooled_prompt_embed"] = torch.from_numpy(np.load(f"{self.opt.prompt_embed_dir}/{uid}_pooled.npy"))
            if "PixArt" in self.opt.pretrained_model_name_or_path:  # PixArt-alpha, PixArt-Sigma
                return_dict["prompt_attention_mask"] = torch.from_numpy(np.load(f"{self.opt.prompt_embed_dir}/{uid}_attention_mask.npy"))

        for key in return_dict.keys():
            assert isinstance(return_dict[key], Tensor), f"Value of the key [{key}] is not a Tensor, but {type(return_dict[key])}."

        return dict(return_dict)

    def _load_png(self, png_bytes: Union[bytes, str], uint16: bool = False) -> Tensor:
        png = np.frombuffer(png_bytes, np.uint8)
        png = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)  # (H, W, C) ndarray in [0, 255] or [0, 65553]

        png = png.astype(np.float32) / (65535. if uint16 else 255.)  # (H, W, C) in [0, 1]
        png[:, :, :3] = png[:, :, :3][..., ::-1]  # BGR -> RGB
        png_tensor = torch.from_numpy(png).nan_to_num_(0.)  # there are nan in GObjaverse gt normal
        return png_tensor.permute(2, 0, 1)  # (C, H, W) in [0, 1]

    def _load_camera_from_json(self, json_bytes: Union[bytes, str]) -> ndarray:
        if isinstance(json_bytes, bytes):
            json_dict = json.loads(json_bytes)
        else:  # BACKUP
            path = os.path.join(self.opt.backup_file_dir, f"{json_bytes}.json")
            with open(path, "r") as f:
                json_dict = json.load(f)

        # In OpenCV convention
        c2w = np.eye(4)  # float64
        c2w[:3, 0] = np.array(json_dict["x"])
        c2w[:3, 1] = np.array(json_dict["y"])
        c2w[:3, 2] = np.array(json_dict["z"])
        c2w[:3, 3] = np.array(json_dict["origin"])
        return c2w

    def _pick_even_view_indices(self, num_views: int = 4) -> List[int]:
        assert 12 % num_views == 0  # `12` for even-view sampling in GObjaverse

        if np.random.rand() < 2/3:
            index0 = np.random.choice(range(24))  # 0~23: 24 views in ele from [5, 30]; hard-coded for GObjaverse
            return [(index0 + (24 // num_views)*i) % 24 for i in range(num_views)]
        else:
            index0 = np.random.choice(range(12))  # 27~38: 12 views in ele from [-5, 5]; hard-coded for GObjaverse
            return [((index0 + (12 // num_views)*i) % 12 + 27) for i in range(num_views)]

    def _pick_random_view_indices(self, num_views: int = 4) -> List[int]:
        assert num_views <= 40  # `40` is hard-coded for GObjaverse

        indices = (set(range(40)) - set([25, 26])) if self.opt.exclude_topdown_views else (set(range(40)))  # `40` is hard-coded for GObjaverse
        return np.random.choice(list(indices), num_views, replace=False).tolist()

    def _check_views_exist(self, sample: Dict[str, Union[str, bytes]], vids: List[int]) -> bool:
        for vid in vids:
            if f"{vid:05d}.png" not in sample:
                return False
            try:
                assert sample[f"{vid:05d}.png"] is not None and sample[f"{vid:05d}.json"] is not None
            except:  # TypeError: a bytes-like object is required, not 'NoneType'; KeyError: '00001.json'
                return False
        return True

    def _check_views_exist_disk(self, uid: str, vids: List[int]) -> bool:
        for vid in vids:
            if not (os.path.exists(os.path.join(self.opt.backup_file_dir, f"{uid}.{vid:05d}.png"))
                and os.path.exists(os.path.join(self.opt.backup_file_dir, f"{uid}.{vid:05d}.json"))):
                return False
        return True
