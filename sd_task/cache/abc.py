from typing import Protocol, TypeVar, Callable

T = TypeVar("T")


class ModelCache(Protocol[T]):
    def load(self, key: str, model_loader: Callable[[], T]) -> T:
        ...
