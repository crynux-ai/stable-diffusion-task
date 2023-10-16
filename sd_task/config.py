from __future__ import annotations

import os
from typing import List, Literal, Tuple, Any, Dict, Type

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_settings import (
    SettingsConfigDict,
    BaseSettings,
    PydanticBaseSettingsSource,
)
from pydantic_settings.main import BaseSettings
import yaml


class YamlSettingsConfigDict(SettingsConfigDict):
    yaml_file: str | None


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    A simple settings source class that loads variables from a YAML file

    Note: slightly adapted version of JsonConfigSettingsSource from docs.
    """

    _yaml_data: Dict[str, Any] | None = None

    # def __init__(self, settings_cls: type[BaseSettings]):
    #     super().__init__(settings_cls)

    @property
    def yaml_data(self) -> Dict[str, Any]:
        if self._yaml_data is None:
            yaml_file = self.config.get("yaml_file")
            if yaml_file is not None and os.path.exists(yaml_file):
                with open(yaml_file, mode="r", encoding="utf-8") as f:
                    self._yaml_data = yaml.safe_load(f)
            else:
                self._yaml_data = {}
        return self._yaml_data

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Any, str, bool]:
        field_value = self.yaml_data.get(field_name)
        return field_value, field_name, False

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        return value

    def __call__(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}

        for field_name, field in self.settings_cls.model_fields.items():
            field_value, field_key, value_is_complex = self.get_field_value(
                field, field_name
            )
            field_value = self.prepare_field_value(
                field_name, field, field_value, value_is_complex
            )
            if field_value is not None:
                d[field_key] = field_value

        return d


class ModelsDirConfig(BaseModel):
    huggingface: str
    external: str


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


class Config(BaseSettings):
    data_dir: DataDirConfig = DataDirConfig(
        models=ModelsDirConfig(
            huggingface="models/huggingface", external="models/external"
        )
    )
    preloaded_models: PreloadedModelsConfig = PreloadedModelsConfig(
        base=[
            ModelConfig(id="runwayml/stable-diffusion-v1-5"),
            ModelConfig(id="emilianJR/chilloutmix_NiPrunedFp32Fix"),
            ModelConfig(id="stabilityai/stable-diffusion-xl-base-1.0", variant="fp16"),
            ModelConfig(
                id="stabilityai/stable-diffusion-xl-refiner-1.0", variant="fp16"
            ),
        ],
        controlnet=[
            ModelConfig(id="lllyasviel/sd-controlnet-canny"),
            ModelConfig(id="lllyasviel/control_v11p_sd15_openpose"),
            ModelConfig(id="diffusers/controlnet-canny-sdxl-1.0"),
        ],
        vae=[],
    )
    proxy: ProxyConfig | None = None

    model_config = YamlSettingsConfigDict(
        env_nested_delimiter="__", yaml_file=os.getenv("SD_TASK_CONFIG", "config.yml"), env_file=".env"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


_default_config: Config | None = None


def get_config() -> Config:
    global _default_config

    if _default_config is None:
        _default_config = Config()  # type: ignore

    return _default_config