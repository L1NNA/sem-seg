from typing import Any, List, Tuple

from utils.config import Config


class BaseClassification:

    def __init__(self, config:Config):
        self.config = config
        self.sources:List[Tuple[str, str]] = []

    def get_sources(self, sources:List[Tuple[str, str]]):
        self.sources = sources

    def classify(self, target:str):
        pass

    def __call__(self, *args: Any, **_: Any) -> Any:
        return self.classify(args[0])