from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Decision = Literal["decompose", "no_decomposition"]


@dataclass
class Subtask:
    id: str
    text: str


@dataclass
class Dependency:
    before: str
    after: str


@dataclass
class Decomposition:
    decision: Decision
    rationale: str = ""
    subtasks: list[Subtask] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)


@dataclass
class TaskExample:
    task: str
    context: str = ""
    decision: Decision = "decompose"
    subtasks: list[str] = field(default_factory=list)
    dependencies: list[tuple[str, str]] = field(default_factory=list)
