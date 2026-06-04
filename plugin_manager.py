from __future__ import annotations

import functools
from typing import Callable, List, Type

_importer_registry: List[Type] = []
_solver_registry: List[Type] = []


def register_importer(cls: Type) -> Type:
    _importer_registry.append(cls)
    return cls


def get_importer_classes() -> List[Type]:
    return list(_importer_registry)


def register_solver_backend(cls: Type) -> Type:
    _solver_registry.append(cls)
    return cls


def get_solver_backend_classes() -> List[Type]:
    return list(_solver_registry)
