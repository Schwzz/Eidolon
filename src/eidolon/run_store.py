"""Single-writer run locks and bounded, asynchronous atomic checkpoint I/O."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

from eidolon.evolution import EvolutionTrainer
from eidolon.experiment_io import write_json


class RunLock:
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.stream = (directory / ".writer.lock").open("a+b")
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise ValueError("training directory is already open for writing; choose another run directory") from error

    def close(self):
        self.stream.close()  # The OS releases the lock even after a process crash.

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def save_snapshot(directory, snapshot):
    # The complete live snapshot is authoritative, including unfinished-generation results.
    write_json(Path(directory) / "live.json", snapshot)
    EvolutionTrainer.from_checkpoint_data(snapshot["trainer"]).save(directory)


class SnapshotWriter:
    """One disk job plus one newest pending snapshot; never evaluates a candidate."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eidolon-save")
        self.future = None
        self.pending = None

    def submit(self, snapshot):
        self.pending = snapshot
        self.poll()

    def poll(self):
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            future.result()  # Surface failed writes instead of claiming progress was saved.
        if self.future is None and self.pending is not None:
            snapshot, self.pending = self.pending, None
            self.future = self.executor.submit(save_snapshot, self.directory, snapshot)

    @property
    def busy(self):
        return self.future is not None or self.pending is not None

    def flush(self):
        while self.busy:
            self.poll()
            if self.future is not None:
                self.future.result()
                self.poll()

    def close(self):
        try:
            self.flush()
        finally:
            self.executor.shutdown(wait=True)
