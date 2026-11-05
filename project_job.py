"""Cancellable file operations with owner-thread publication; no Tk references."""
import queue
import threading
import traceback

from main import StitchingCancelled
from stitching_task import TaskState


def _run(operation, events, cancel):
    try:
        result = operation(cancel.is_set, lambda message: events.put(('log', message)))
        events.put(('succeeded', result))
    except StitchingCancelled:
        events.put(('cancelled', None))
    except Exception:
        events.put(('failed', traceback.format_exc()))


class ProjectJob:
    def __init__(self):
        self.state = TaskState.IDLE
        self.kind = None
        self.result = self.error = self.thread = None
        self._events = queue.Queue()
        self._cancel = threading.Event()
        self._terminal = None

    @property
    def busy(self):
        return self.state in (TaskState.RUNNING, TaskState.CANCELLING)

    def start(self, kind, operation):
        if self.busy:
            raise RuntimeError('A project operation is already running')
        self.kind = kind
        self.result = self.error = self._terminal = None
        self._events = queue.Queue()
        self._cancel = threading.Event()
        self.state = TaskState.RUNNING
        self.thread = threading.Thread(target=_run, args=(operation, self._events, self._cancel), daemon=True)
        try:
            self.thread.start()
        except Exception as error:
            self.state = TaskState.FAILED
            self.error = str(error)
            raise

    def cancel(self):
        if self.busy:
            self._cancel.set()
            self.state = TaskState.CANCELLING

    def close(self):
        self.cancel()
        # A late worker result goes to its detached queue, not a closed GUI.
        self._events = queue.Queue()
        self.result = self._terminal = None

    def poll(self):
        logs = []
        while True:
            try:
                kind, value = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == 'log':
                logs.append(value)
            else:
                self._terminal = (kind, value)
        if self.busy and self._terminal is not None and not self.thread.is_alive():
            kind, value = self._terminal
            # Opening discards queued success after accepted cancellation. A
            # save that already atomically committed must report the saved file.
            if kind == 'cancelled' or (self.kind == 'open' and self._cancel.is_set()):
                self.state = TaskState.CANCELLED
            elif kind == 'succeeded':
                self.state = TaskState.SUCCEEDED
                self.result = value
            else:
                self.state = TaskState.FAILED
                self.error = value
            self._terminal = None
        return logs
