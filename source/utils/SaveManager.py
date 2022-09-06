import math
from pathlib import Path
import time
import uuid

import torch


class Savable:
    """Interface used by SaveManager."""

    def save(self) -> dict:
        """Create dict with class state."""
        raise NotImplementedError

    def restore(self, state) -> None:
        """Restore class state."""
        raise NotImplementedError

class SaveManager():
    """
    Save data via torch.save. Also manages save files based on policy.

    Attributes:
        path : Path
            Where to create save files.
        origin : Savable
            Class instance that inherits from Savable.
        min_saves : int
            Minimal number of files. Won't be deleted.
        max_saves : int
            Maximum number if files.
        max_size : int [MB]
            Maximum size of saved files.
        time_interval : int [s]
            Time in seconds that has to past before another save will be created.
    """

    def __init__(self, path: Path, origin: Savable, min_saves: int, max_saves: int, max_size: int, time_interval: int) -> None:
        self.path = path
        self.origin = origin
        self.min_saves = min_saves
        self.max_saves = max_saves
        self.max_size = max_size
        self.time_interval = time_interval

        self.path.mkdir(parents=True, exist_ok=True)

    def save_exists(self, name : str = None) -> bool:
        """
        Check if save exists.

        Arguments:
            name : str = None
                Name of the file to look for. If none, check if any save exists.
        """

        if name:
            return Path(self.path, f"{name}").exists()
        else:
            paths = list(self.path.glob('*.tar'))
            return True if len(paths)>0 else False

    def save(self, force: bool = False) -> None:
        """
        Save state of origin.

        Arguments:
            force: bool = False
                Ignore time interval.
        """

        if not force and self._since_last_save() < self.time_interval:
            return

        self._remove_old()

        to_save = self.origin.save()
        torch.save(to_save, Path(self.path, f"{uuid.uuid4().hex}.tar"))

    def load(self, name: str = None) -> None:
        """
        Restore state from save.

        Arguments:
            name: str = None
                Name of the file to restore. If none, choose most up to date.
        """
        if not self.save_exists(name):
            return

        if name:
            save = torch.load(Path(self.path, f"{name}"))
        else:
            a = self._find_latest().absolute()
            save = torch.load(a)

        self.origin.restore(save)

    def _since_last_save(self) -> float:
        """Return time in seconds since last save."""

        if self.save_exists():
            return time.time() - self._find_latest().stat().st_mtime
        else:
            return math.inf

    def _find_latest(self) -> Path:
        """Return path of most up to date save."""

        paths = self.path.glob('*.tar')
        return sorted(paths, key=lambda x: x.stat().st_mtime)[-1]

    def _remove_old(self) -> None:
        """Remove files based on policy."""

        paths = self.path.glob('*.tar')
        paths = sorted(paths, key=lambda x: x.stat().st_mtime)

        if len(paths) <= self.min_saves:
            return

        if len(paths) > self.max_saves:
            amount_to_delete = len(paths) - self.max_saves
            files_to_delete = paths[:amount_to_delete]
            paths = paths[amount_to_delete:]
            self._delete_files(files_to_delete)

        current_size = sum([f.stat().st_size for f in paths])/(1<<20)
        while len(paths) > self.min_saves and current_size > self.max_size:
            file_to_delete = paths.pop(0)
            current_size -= file_to_delete.stat().st_size/(1<<20)
            self._delete_files([file_to_delete])

    def _delete_files(self, files: list[Path]) -> None:
        """Delete files from list."""

        for f in files:
            f.unlink()
