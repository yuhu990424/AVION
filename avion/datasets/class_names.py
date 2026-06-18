from __future__ import annotations

import re


def split_camel_case(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    return value


def canonicalize_class_name(value: str) -> str:
    value = split_camel_case(value)
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def prompt_name(value: str) -> str:
    return canonicalize_class_name(value)

