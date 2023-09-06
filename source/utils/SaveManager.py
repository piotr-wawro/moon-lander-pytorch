from datetime import datetime
from pathlib import Path
from time import time

import torch


class Savable:
  def save(self) -> dict:
    raise NotImplementedError

  def restore(self, state) -> None:
    raise NotImplementedError

class SaveManager():
  def __init__(self, path: Path, obj: Savable, min_saves: int, max_saves: int, max_mb_size: int, time_interval: int) -> None:
    self.path = path
    self.obj = obj
    self.min_saves = min_saves
    self.max_saves = max_saves
    self.max_mb_size = max_mb_size
    self.time_interval = time_interval
    self.last_save = 0

    self.path.mkdir(parents=True, exist_ok=True)

  def save(self, force: bool = False) -> None:
    if not force and self._since_last_save() < self.time_interval:
      return

    self._remove_old()
    self.last_save = time()

    to_save = self.obj.save()
    torch.save(to_save, Path(self.path, f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.tar"))

  def load(self, name: str = None) -> None:
    if not self.save_exist(name):
      return

    if name:
      save = torch.load(Path(self.path, f"{name}"))
    else:
      a = self._find_latest().absolute()
      save = torch.load(a)
      print(f"Loaded {a}")

    self.obj.restore(save)

  def save_exist(self, name : str = None) -> bool:
    if name:
      return Path(self.path, f"{name}").exists()
    else:
      paths = list(self.path.glob('*.tar'))
      return True if len(paths) > 0 else False

  def _since_last_save(self) -> float:
    return time() - self.last_save

  def _find_latest(self) -> Path:
    paths = self.path.glob('*.tar')
    return max(paths, key=lambda x: x.stat().st_mtime)

  def _remove_old(self) -> None:
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
    while len(paths) > self.min_saves and current_size > self.max_mb_size:
      file_to_delete = paths.pop(0)
      current_size -= file_to_delete.stat().st_size/(1<<20)
      self._delete_files([file_to_delete])

  def _delete_files(self, files: list[Path]) -> None:
    for f in files:
      f.unlink()
