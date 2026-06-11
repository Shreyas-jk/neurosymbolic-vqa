"""Top-level vision orchestrator: image → SceneGraph.

Pipeline:
  1. OWL-ViT zero-shot detection over OBJECT_VOCAB → boxes + scores + labels.
  2. Score-threshold filter, then NMS via torchvision.ops.nms (per-class).
  3. For each surviving box, crop and route to CLIP attribute classifier.
  4. Pairwise spatial relations via geometric rules over normalized bboxes.
  5. Wrap into a validated SceneGraph (pydantic does the bbox / id checks).

The extractor is a thin orchestrator — the heavy lifting lives in `models`,
`attribute_classifier`, and `spatial_relations`. Each stage is independently
callable, which keeps the unit-test surface small.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torchvision.ops as ops
from PIL import Image

from scene_extractor import attribute_classifier
from scene_extractor import spatial_relations as _spatial
from scene_extractor.config import (
    CLIP_MODEL_ID,
    DETECTION_THRESHOLD,
    DETECTOR_MODEL_ID,
    NMS_IOU_THRESHOLD,
    OBJECT_VOCAB,
)
from scene_extractor.models import get_detector, get_device
from scene_extractor.schema import BoundingBox, SceneGraph, SceneObject


class ModelDownloadError(RuntimeError):
    """Raised when a HuggingFace model fails to load (cache miss + no network)."""


class SceneExtractor:
    """Vision pipeline producing a SceneGraph from an image.

    Parameters
    ----------
    object_vocab:
        Candidate categories shown to OWL-ViT as text queries. Defaults to the
        CLEVR-friendly OBJECT_VOCAB from config; pass a domain-specific list to
        sharpen detection on out-of-distribution scenes.
    detection_threshold:
        Score floor for OWL-ViT detections. OWL-ViT scores run low; 0.1 is the
        documented sweet spot for the base model.
    nms_iou:
        IoU above which the higher-scoring box wins during NMS.
    detector_model_id / clip_model_id:
        HuggingFace IDs. Constructor params so the eval harness can A/B without
        editing config.
    """

    def __init__(
        self,
        *,
        object_vocab: Sequence[str] = OBJECT_VOCAB,
        detection_threshold: float = DETECTION_THRESHOLD,
        nms_iou: float = NMS_IOU_THRESHOLD,
        detector_model_id: str = DETECTOR_MODEL_ID,
        clip_model_id: str = CLIP_MODEL_ID,
    ) -> None:
        self.object_vocab = tuple(object_vocab)
        self.detection_threshold = detection_threshold
        self.nms_iou = nms_iou
        self.detector_model_id = detector_model_id
        self.clip_model_id = clip_model_id

    def extract(self, image: Union[str, Path, Image.Image]) -> SceneGraph:
        start = time.perf_counter()

        pil = self._load_image(image)
        image_path = str(image) if not isinstance(image, Image.Image) else None

        objects = self._detect_and_classify(pil)
        relations = _spatial.compute(objects)

        return SceneGraph(
            image_path=image_path,
            objects=objects,
            relations=relations,
            extraction_time_ms=(time.perf_counter() - start) * 1000.0,
            model_versions={
                "detector": self.detector_model_id,
                "clip": self.clip_model_id,
            },
        )

    @staticmethod
    def _load_image(image: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            pil = image
        else:
            pil = Image.open(image)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return pil

    def _detect_and_classify(self, pil: Image.Image) -> list[SceneObject]:
        try:
            processor, model = get_detector(self.detector_model_id)
        except Exception as exc:
            raise ModelDownloadError(
                f"Failed to load detector {self.detector_model_id!r}: {exc}. "
                "Run scripts/download_models.sh or check network/HF cache."
            ) from exc

        device = get_device()
        text_queries = [f"a photo of a {c}" for c in self.object_vocab]
        inputs = processor(text=[text_queries], images=pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)

        target_sizes = torch.tensor([(pil.height, pil.width)], device=device)
        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.detection_threshold,
        )[0]

        boxes = results["boxes"].detach().cpu()
        scores = results["scores"].detach().cpu()
        labels = results["labels"].detach().cpu()

        if boxes.numel() == 0:
            return []

        # Per-class NMS so a single object doesn't keep two boxes from rival
        # category labels.
        keep_mask = torch.zeros(len(boxes), dtype=torch.bool)
        for cls in torch.unique(labels):
            cls_mask = labels == cls
            idxs = torch.nonzero(cls_mask, as_tuple=False).flatten()
            if len(idxs) == 0:
                continue
            kept = ops.nms(boxes[idxs], scores[idxs], self.nms_iou)
            keep_mask[idxs[kept]] = True

        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        labels = labels[keep_mask]

        objects: list[SceneObject] = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box.tolist()
            bbox = self._to_normalized_bbox(x1, y1, x2, y2, pil.width, pil.height)
            if bbox is None:
                continue
            category = self.object_vocab[int(label.item())]
            crop = pil.crop((int(max(0, x1)), int(max(0, y1)), int(min(pil.width, x2)), int(min(pil.height, y2))))
            if crop.width == 0 or crop.height == 0:
                continue
            try:
                attrs, conf = attribute_classifier.classify(crop)
            except Exception as exc:
                raise ModelDownloadError(
                    f"Failed to run CLIP attribute classifier: {exc}"
                ) from exc
            objects.append(
                SceneObject(
                    id=f"obj_{i}",
                    category=category,
                    confidence=float(score.item()),
                    bbox=bbox,
                    attributes=attrs,
                    attribute_confidences=conf,
                )
            )
        return objects

    @staticmethod
    def _to_normalized_bbox(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        height: int,
    ) -> Optional[BoundingBox]:
        nx1 = max(0.0, min(1.0, x1 / width))
        ny1 = max(0.0, min(1.0, y1 / height))
        nx2 = max(0.0, min(1.0, x2 / width))
        ny2 = max(0.0, min(1.0, y2 / height))
        # Skip degenerate boxes (e.g., post-clip collapse).
        if not (nx1 < nx2 and ny1 < ny2):
            return None
        return BoundingBox(x1=nx1, y1=ny1, x2=nx2, y2=ny2)
