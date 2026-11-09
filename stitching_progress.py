"""Immutable counts for the current stage, never an estimated overall percent."""
from dataclasses import dataclass


@dataclass(frozen=True)
class StageProgress:
    stage: str
    completed: int = 0
    total: int | None = None

    def __post_init__(self):
        if self.stage not in ('load', 'match', 'solve', 'blend'):
            raise ValueError('Unknown stitching stage')
        if self.completed < 0 or (self.total is not None and
                                   (self.total <= 0 or self.completed > self.total)):
            raise ValueError('Invalid stage counts')

    @property
    def fraction(self):
        return None if self.total is None else self.completed / self.total
