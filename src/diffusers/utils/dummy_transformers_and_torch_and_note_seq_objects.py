# This file is autogenerated by the command `make fix-copies`, do not edit.
from ..utils import DummyObject, requires_backends


class SpectrogramDiffusionPipeline(metaclass=DummyObject):
    _backends = ["transformers", "torch", "note_seq"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["transformers", "torch", "note_seq"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["transformers", "torch", "note_seq"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["transformers", "torch", "note_seq"])
