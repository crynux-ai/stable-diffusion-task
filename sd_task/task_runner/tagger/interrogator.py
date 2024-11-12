import json
from abc import ABC, abstractmethod
from typing import Tuple, Dict

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from pandas import read_csv
from PIL import Image

from sd_task import utils

from . import imutils


def get_onnx_providers():
    providers = []
    if utils.get_accelerator() == "cuda":
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "cudnn_conv_algo_search": "DEFAULT",
                    "use_tf32": 0,
                    "device_id": torch.cuda.current_device(),
                },
            )
        )
    providers.append("CPUExecutionProvider")
    return providers


class Interrogator(ABC):
    def __init__(self, name: str):
        self.name = name

        self.model = None
        self.tags = None

    @abstractmethod
    def load(self, cache_dir: str): ...

    @abstractmethod
    def interrogate(self, image: Image.Image) -> Tuple[Dict[str, float], Dict[str, float]]: ...


class WaifuDiffusionInterrogator(Interrogator):
    def __init__(
        self,
        name: str,
        repo_id: str,
        model_path: str = "model.onnx",
        tags_path: str = "selected_tags.csv",
    ) -> None:
        super().__init__(name=name)
        self.repo_id = repo_id
        self.model_path = model_path
        self.tags_path = tags_path

    def download(self, cache_dir: str):
        model_path = hf_hub_download(
            repo_id=self.repo_id, filename=self.model_path, cache_dir=cache_dir
        )
        tags_path = hf_hub_download(
            repo_id=self.repo_id, filename=self.tags_path, cache_dir=cache_dir
        )
        return model_path, tags_path

    def load(self, cache_dir: str):
        model_path, tags_path = self.download(cache_dir=cache_dir)
        providers = get_onnx_providers()
        self.model = ort.InferenceSession(model_path, providers=providers)
        self.tags = read_csv(tags_path)

    def interrogate(self, image: Image.Image):
        assert self.model is not None
        assert self.tags is not None

        # convert an image to fit the model
        _, height, _, _ = self.model.get_inputs()[0].shape

        # alpha to white
        image = imutils.fill_transparent(image)

        img = np.asarray(image)  # type: ignore
        # PIL RGB to OpenCV BGR
        img = img[:, :, ::-1]

        img = imutils.make_square(img, height)
        img = imutils.smart_resize(img, height)
        img = img.astype(np.float32)
        img = np.expand_dims(img, 0)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        confidences = self.model.run([label_name], {input_name: img})[0]

        tags = self.tags[:][["name"]]
        tags["confidences"] = confidences[0]

        # first 4 items are for rating (general, sensitive, questionable,
        # explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)

        return ratings, tags


class MLDanbooruInterrogator(Interrogator):
    def __init__(
        self, name: str, repo_id: str, model_path: str, tags_path: str = "classes.json"
    ):
        super().__init__(name)
        self.repo_id = repo_id
        self.model_path = model_path
        self.tags_path = tags_path

    def download(self, cache_dir: str):
        model_path = hf_hub_download(
            repo_id=self.repo_id, filename=self.model_path, cache_dir=cache_dir
        )
        tags_path = hf_hub_download(
            repo_id=self.repo_id, filename=self.tags_path, cache_dir=cache_dir
        )
        return model_path, tags_path

    def load(self, cache_dir: str):
        model_path, tags_path = self.download(cache_dir=cache_dir)

        providers = get_onnx_providers()
        self.model = ort.InferenceSession(model_path, providers=providers)

        with open(tags_path, "r", encoding="utf-8") as f:
            self.tags = json.load(f)

    def interrogate(self, image: Image.Image):
        assert self.model is not None
        assert self.tags is not None

        image = imutils.fill_transparent(image)
        image = imutils.resize(image, 448)  # TODO CUSTOMIZE

        x = np.asarray(image, dtype=np.float32) / 255
        # HWC -> 1CHW
        x = x.transpose((2, 0, 1))
        x = np.expand_dims(x, 0)

        input_ = self.model.get_inputs()[0]
        output = self.model.get_outputs()[0]
        # evaluate model
        (y,) = self.model.run([output.name], {input_.name: x})

        # Softmax
        y = 1 / (1 + np.exp(-y))

        tags = {tag: float(conf) for tag, conf in zip(self.tags, y.flatten())}
        return {}, tags
