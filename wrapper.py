"""
external/FiLo/wrapper.py

FiLo wrapper for anomalib external model registry.

Implements:
- zero-shot FiLo inference
- optional Grounding DINO guidance
- custom prompts for non-benchmark classes such as "weld seam"

Important:
- This wrapper expects test_model.py to call:
    model.predict(image_tensor, image_path=path_str)
  If image_path is not provided, Grounding DINO is skipped automatically.
"""

import copy
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

_GDINO_ROOT = _HERE / "models" / "GroundingDINO"
sys.path.insert(0, str(_GDINO_ROOT))

from models.FiLo import FiLo, cls_map, positions_list
import models.GroundingDINO.groundingdino.datasets.transforms as T  # noqa: E402
from models.GroundingDINO.groundingdino.models import build_model  # noqa: E402
from models.GroundingDINO.groundingdino.util.slconfig import SLConfig  # noqa: E402
from models.GroundingDINO.groundingdino.util.utils import (  # noqa: E402
    clean_state_dict,
    get_phrases_from_posmap,
)


class FiLoWrapper:
    def __init__(
        self,
        class_name: str = "weld seam",
        checkpoint_path: str | None = None,
        groundingdino_config_path: str | None = None,
        groundingdino_checkpoint_path: str | None = None,
        use_grounding: bool = False,
        clip_model: str = "ViT-L-14-336",
        clip_pretrained: str = "openai",
        image_size: int = 518,
        features_list: list[int] | None = None,
        n_ctx: int = 12,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        area_threshold: float = 0.7,
        device: str | None = None,
    ):
        self.class_name = class_name.replace("_", " ")
        self.image_size = image_size
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.area_threshold = area_threshold
        self.use_grounding = use_grounding

        if features_list is None:
            features_list = [6, 12, 18, 24]

        self.device = self._select_device(device)

        self.location = {
            "top left": [(0, 0), (172, 172)],
            "top": [(173, 0), (344, 172)],
            "top right": [(345, 0), (517, 172)],
            "left": [(0, 173), (172, 344)],
            "center": [(173, 173), (344, 344)],
            "right": [(345, 173), (517, 344)],
            "bottom left": [(0, 345), (172, 517)],
            "bottom": [(173, 345), (344, 517)],
            "bottom right": [(345, 345), (517, 517)],
        }

        self.anomaly_status_general = [
            "anomaly",
            "damage",
            "broken",
            "defect",
            "contamination",
        ]

        self.custom_anomaly_detail = {
            "weld seam": [
                "porosity",
                "crack",
                "undercut",
                "lack of fusion",
                "spatter",
                "misalignment",
                "burn through",
                "surface contamination",
                "irregular bead",
            ],
        }

        if self.class_name not in cls_map:
            cls_map[self.class_name] = self.class_name

        args = SimpleNamespace(
            clip_model=clip_model,
            clip_pretrained=clip_pretrained,
            image_size=image_size,
            features_list=features_list,
            n_ctx=n_ctx,
            device=self.device,
        )

        self.model = FiLo([self.class_name], args, self.device).to(self.device)
        self.model.eval()

        if checkpoint_path is None:
            raise ValueError("FiLoWrapper requires checkpoint_path for the FiLo checkpoint.")

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        filo_state = ckpt["filo"] if isinstance(ckpt, dict) and "filo" in ckpt else ckpt
        self.model.load_state_dict(filo_state, strict=False)
        self.model.eval()

        self.gaussian_filter = torchvision.transforms.GaussianBlur(3, 4.0)

        self.grounding_model = None
        if self.use_grounding:
            if not groundingdino_config_path or not groundingdino_checkpoint_path:
                raise ValueError(
                    "use_grounding=True requires groundingdino_config_path and groundingdino_checkpoint_path."
                )
            self.grounding_model = self._load_grounding_model(
                groundingdino_config_path,
                groundingdino_checkpoint_path,
            )

    def _select_device(self, requested: str | None) -> str:
        if requested is not None:
            requested = requested.lower()
            if requested == "cuda" and torch.cuda.is_available():
                return "cuda"
            if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            if requested == "cpu":
                return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resize_pad_input(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize preserving aspect ratio, then center-pad to self.image_size.
        Expects [3, H, W], returns [3, image_size, image_size].
        """
        if image_tensor.ndim != 3:
            raise ValueError(f"Expected image tensor [3,H,W], got {tuple(image_tensor.shape)}")

        _, h, w = image_tensor.shape
        target = self.image_size

        scale = min(target / h, target / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        x = image_tensor.unsqueeze(0)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        pad_h = target - new_h
        pad_w = target - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
        return x.squeeze(0)

    def _load_grounding_model(self, config_path: str, checkpoint_path: str):
        args = SLConfig.fromfile(config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(clean_state_dict(state), strict=False)
        model.eval()
        return model.to(self.device)

    def _load_image_for_dino(self, image_path: str):
        image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose(
            [
                T.RandomResize([self.image_size, self.image_size], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image_pil, None)
        return image_pil, image_tensor

    def _build_text_prompt(self) -> str:
        if self.class_name in self.custom_anomaly_detail:
            detail = self.custom_anomaly_detail[self.class_name]
        else:
            detail = [
                "surface defect",
                "crack",
                "scratch",
                "contamination",
                "deformation",
            ]
        return " . ".join(self.anomaly_status_general + detail)

    def _get_grounding_output(self, image: torch.Tensor, caption: str, with_logits: bool = True):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."

        model = self.grounding_model.to(self.device)
        image = image.to(self.device)

        with torch.no_grad():
            outputs = model(image[None], captions=[caption])

        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        boxes_area = boxes_filt[:, 2] * boxes_filt[:, 3]

        filt_mask = torch.bitwise_and(
            logits_filt.max(dim=1)[0] > self.box_threshold,
            boxes_area < self.area_threshold,
        )

        if torch.sum(filt_mask) == 0:
            filt_mask = torch.argmax(logits_filt.max(dim=1)[0])
            logits_filt = logits_filt[filt_mask].unsqueeze(0)
            boxes_filt = boxes_filt[filt_mask].unsqueeze(0)
        else:
            logits_filt = logits_filt[filt_mask]
            boxes_filt = boxes_filt[filt_mask]

        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)

        pred_phrases = []
        boxes_out = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenizer)
            if with_logits:
                pred_phrase = pred_phrase + f"({str(logit.max().item())[:4]})"
            pred_phrases.append(pred_phrase)
            boxes_out.append(box)

        if len(boxes_out) == 0:
            return None, []

        boxes_out = torch.stack(boxes_out, dim=0)
        return boxes_out, pred_phrases

    def _convert_boxes_to_xyxy(self, boxes_filt: torch.Tensor) -> torch.Tensor:
        boxes_filt_copy = copy.deepcopy(boxes_filt)
        for i in range(boxes_filt_copy.size(0)):
            boxes_filt_copy[i] = boxes_filt_copy[i] * torch.tensor(
                [self.image_size, self.image_size, self.image_size, self.image_size]
            )
            boxes_filt_copy[i][:2] -= boxes_filt_copy[i][2:] / 2
            boxes_filt_copy[i][2:] += boxes_filt_copy[i][:2]
        return boxes_filt_copy.cpu()

    def _boxes_to_positions(self, boxes_filt: torch.Tensor, pred_phrases: list[str]) -> list[str]:
        if boxes_filt is None or len(pred_phrases) == 0:
            return []

        position = []
        max_box = None
        max_pred = -1.0

        for i in range(boxes_filt.size(0)):
            phrase = pred_phrases[i]
            m = re.search(r"\((.*?)\)", phrase)
            score = float(m.group(1)) if m else 0.0
            if score >= max_pred:
                max_box = boxes_filt[i]
                max_pred = score

        if max_box is not None:
            center = ((max_box[0] + max_box[2]) / 2, (max_box[1] + max_box[3]) / 2)
        else:
            center = (self.image_size / 2, self.image_size / 2)

        for region, ((x1, y1), (x2, y2)) in self.location.items():
            if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                position.append(region)
                break

        return position

    def _apply_grounding_gate(self, anomaly_map: torch.Tensor, boxes_filt: torch.Tensor | None) -> torch.Tensor:
        if boxes_filt is None or boxes_filt.numel() == 0:
            return anomaly_map

        anomaly_score_copy = anomaly_map.clone()
        for rect in boxes_filt:
            left_top_x = max(0, int(rect[0].item()))
            left_top_y = max(0, int(rect[1].item()))
            right_bottom_x = min(self.image_size, int(rect[2].item()))
            right_bottom_y = min(self.image_size, int(rect[3].item()))
            anomaly_score_copy[:, :, left_top_y:right_bottom_y, left_top_x:right_bottom_x] = 1

        return torch.where(anomaly_score_copy == 1, anomaly_map, anomaly_map * 0.7)

    def fit(self, dataloader=None):
        pass

    def eval(self):
        self.model.eval()
        if self.grounding_model is not None:
            self.grounding_model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    def predict(self, image_tensor: torch.Tensor, image_path: str | None = None) -> tuple:
        """
        image_tensor: [3, H, W] float tensor already normalized by anomalib dataloader
        image_path: original file path, required for Grounding DINO path-based loading

        Returns:
            anomaly_map: np.ndarray [H, W]
            score: float
        """
        boxes_filt_xyxy = None
        positions = []

        image_tensor = self._resize_pad_input(image_tensor)

        if self.use_grounding and self.grounding_model is not None and image_path is not None:
            try:
                _, image_dino = self._load_image_for_dino(image_path)
                text_prompt = self._build_text_prompt()
                boxes_filt, pred_phrases = self._get_grounding_output(image_dino, text_prompt)
                if boxes_filt is not None:
                    boxes_filt_xyxy = self._convert_boxes_to_xyxy(boxes_filt)
                    positions = self._boxes_to_positions(boxes_filt_xyxy, pred_phrases)
            except Exception:
                boxes_filt_xyxy = None
                positions = []

        items = {
            "img": image_tensor.unsqueeze(0).to(self.device),
            "cls_name": [self.class_name],
        }

        with torch.no_grad():
            text_probs, anomaly_maps = self.model(
                items,
                with_adapter=True,
                positions=positions,
            )

            processed_maps = []
            for amap in anomaly_maps:
                proc = self.gaussian_filter((amap[:, 1, :, :] - amap[:, 0, :, :] + 1) / 2)
                processed_maps.append(proc)

            anomaly_map = torch.mean(torch.stack(processed_maps, dim=0), dim=0).unsqueeze(1)
            score = (text_probs.flatten()[1].item() + anomaly_map.max().item()) / 2.0
            anomaly_map = self._apply_grounding_gate(anomaly_map, boxes_filt_xyxy)

        return anomaly_map.detach().cpu().numpy().reshape(self.image_size, self.image_size), float(score)