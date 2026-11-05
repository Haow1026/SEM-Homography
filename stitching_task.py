"""Task lifecycle and scan-order mapping, independent of Tk widgets.

The worker only writes to a queue. poll() and every public state mutation run on
its owner (GUI) thread, so closing a window never schedules Tk work from a worker.
"""
from dataclasses import dataclass
from enum import Enum
import os
import queue
import re
import threading
import traceback

from main import SEMStitcher, StitchingCancelled


class TaskState(str, Enum):
    IDLE = 'idle'
    READY = 'ready'
    RUNNING = 'running'
    CANCELLING = 'cancelling'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


def natural_sort(paths):
    def key(path):
        return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                     for part in re.split(r'(\d+)', os.path.basename(path)))
    return sorted(paths, key=key)


@dataclass(frozen=True)
class StitchRequest:
    """An immutable snapshot of files in acquisition order and scan settings."""
    paths: tuple
    grid_shape: tuple | None = None
    serpentine: bool = False
    first_row_rtl: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'paths', tuple(os.fspath(path) for path in self.paths))
        if len(self.paths) < 2:
            raise ValueError('At least two images are required')
        if self.grid_shape is not None:
            object.__setattr__(self, 'grid_shape', tuple(self.grid_shape))
            rows, cols = self.grid_shape
            if (type(rows) is not int or type(cols) is not int or rows < 1 or cols < 1
                    or rows * cols != len(self.paths)):
                raise ValueError('Grid dimensions must be positive and match the image count')
        elif self.serpentine or self.first_row_rtl:
            raise ValueError('Scan direction settings require a grid')

    def layout_indices(self):
        """Physical rows, top to bottom; each row lists left-to-right file indices."""
        if self.grid_shape is None:
            return []
        rows, cols = self.grid_shape
        layout = []
        for row in range(rows):
            indices = list(range(row * cols, (row + 1) * cols))
            if self.first_row_rtl ^ (self.serpentine and row % 2 == 1):
                indices.reverse()
            layout.append(indices)
        return layout

    def algorithm_inputs(self):
        if self.grid_shape is None:
            return self.paths, {}
        # Normalize every scan pattern into physical row-major order, allowing
        # the existing cardinal-neighbour matcher to handle all four patterns.
        paths = tuple(self.paths[index] for row in self.layout_indices() for index in row)
        return paths, {'grid_shape': self.grid_shape, 'serpentine': False}


class StitchTask:
    def __init__(self, stitcher_factory=SEMStitcher):
        self.state = TaskState.IDLE
        self.result = None
        self.inspection = None
        self.metrics = {}
        self.error = None
        self.request = None
        self.thread = None
        self._factory = stitcher_factory
        self._events = queue.Queue()
        self._cancel = threading.Event()
        self._terminal = None

    @property
    def busy(self):
        return self.state in (TaskState.RUNNING, TaskState.CANCELLING)

    def reset(self, has_inputs=False):
        if self.busy:
            raise RuntimeError('Wait for the current task to finish')
        self.result = self.error = self.request = None
        self.inspection = None
        self.metrics = {}
        self.state = TaskState.READY if has_inputs else TaskState.IDLE

    def start(self, request):
        if self.busy:
            raise RuntimeError('A task is already running')
        self.request = request
        self.result = self.error = self._terminal = None
        self.inspection = None
        self.metrics = {}
        self._events = queue.Queue()
        self._cancel = threading.Event()
        self.state = TaskState.RUNNING
        self.thread = threading.Thread(target=self._run, args=(request,), daemon=True)
        try:
            self.thread.start()
        except Exception as error:
            self.error = str(error)
            self.state = TaskState.FAILED
            raise

    def cancel(self):
        if self.busy:
            self._cancel.set()
            self.state = TaskState.CANCELLING

    def _run(self, request):
        try:
            stitcher = self._factory()
            stitcher.set_cancel_callback(self._cancel.is_set)
            stitcher.set_progress_callback(lambda message: self._events.put(('log', message)))
            paths, options = request.algorithm_inputs()
            result = stitcher.stitch_sequence(paths, **options)
            inspection = getattr(stitcher, 'last_inspection', None)
            if inspection is not None and request.grid_shape is not None:
                inspection = inspection.reordered([index for row in request.layout_indices() for index in row])
            self._events.put(('succeeded', (result, inspection,
                                            dict(getattr(stitcher, 'last_metrics', {})))))
        except StitchingCancelled:
            self._events.put(('cancelled', None))
        except Exception:
            self._events.put(('failed', traceback.format_exc()))

    def poll(self):
        """Drain logs; apply a terminal outcome only after the worker exits."""
        logs = []
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == 'log':
                logs.append(payload)
            else:
                self._terminal = (kind, payload)
        if self.busy and self._terminal is not None and not self.thread.is_alive():
            kind, payload = self._terminal
            # A cancellation accepted just before completion still discards the
            # result, including when a noninterruptible OpenCV call just ended.
            if self._cancel.is_set() or kind == 'cancelled':
                self.state = TaskState.CANCELLED
            elif kind == 'succeeded':
                self.result, self.inspection, self.metrics = payload
                self.state = TaskState.SUCCEEDED
            else:
                self.error = payload
                self.state = TaskState.FAILED
            self._terminal = None
        return logs
