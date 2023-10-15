from __future__ import annotations

import os
from typing import List, Literal

from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict
from pydantic_settings_yaml import YamlBaseSettings


class ModelsDirConfig(BaseModel):
    huggingface: str
    lora: str
    textual_inversion: str


class DataDirConfig(BaseModel):
    models: ModelsDirConfig


class ModelConfig(BaseModel):
    id: str
    variant: Literal["fp16", "ema"] | None = None


class PreloadedModelsConfig(BaseModel):
    base: List[ModelConfig] | None 
    controlnet: List[ModelConfig] | None
    vae: List[ModelConfig] | None


class ProxyConfig(BaseModel):
    host: str = ""
    port: int = 8080
    username: str = ""
    password: str = ""


class Config(YamlBaseSettings):
    data_dir: DataDirConfig = DataDirConfig(
        models=ModelsDirConfig(
            huggingface="models/huggingface",
            lora="models/lora",
            textual_inversion="models/textual_inversion"
        )
    )
    preloaded_models: PreloadedModelsConfig = PreloadedModelsConfig(
        base=[
            ModelConfig(id="runwayml/stable-diffusion-v1-5"),
            ModelConfig(id="emilianJR/chilloutmix_NiPrunedFp32Fix"),
            ModelConfig(id="stabilityai/stable-diffusion-xl-base-1.0", variant="fp16"),
            ModelConfig(id="stabilityai/stable-diffusion-xl-refiner-1.0", variant="fp16"),
        ],
        controlnet=[
            ModelConfig(id="lllyasviel/sd-controlnet-canny"),
            ModelConfig(id="lllyasviel/control_v11p_sd15_openpose"),
            ModelConfig(id="diffusers/controlnet-canny-sdxl-1.0"),
        ],
        vae=[],
    )
    proxy: ProxyConfig | None = None

    model_config = SettingsConfigDict(
        yaml_file=os.getenv("SD_TASK_CONFIG", "config.yml"),  # type: ignore
    )


_default_config: Config | None = None


def get_config() -> Config:
    global _default_config

    if _default_config is None:
        _default_config = Config()  # type: ignore

    return _default_config
