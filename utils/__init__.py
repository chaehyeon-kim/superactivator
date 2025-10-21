"""Utilities package setup."""

import builtins
import os
from pathlib import Path
from typing import Any

try:
	import numpy as np
except ImportError:  # pragma: no cover - optional dependency
	np = None

try:
	import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
	pd = None

try:
	import torch
except ImportError:  # pragma: no cover - optional dependency
	torch = None

try:
	from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
	Image = None

# Resolve the repository root assuming this package lives directly under it.
REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_path(*relative_segments: Any) -> Path:
	"""Join the provided segments onto the repository root path."""
	return REPO_ROOT.joinpath(*relative_segments)


def data_path(dataset_name: str, *relative_segments: Any) -> Path:
	"""Shortcut for repository Data directory lookups."""
	return repo_path("Data", dataset_name, *relative_segments)


def figs_path(dataset_name: str, *relative_segments: Any) -> Path:
	"""Shortcut for repository Figs directory outputs."""
	return repo_path("Figs", dataset_name, *relative_segments)


def repo_file(relative_path: str) -> Path:
	"""Resolve a string path relative to the repository root."""
	path_obj = Path(relative_path)
	if path_obj.is_absolute():
		return path_obj

	cleaned = relative_path.strip()
	while cleaned.startswith("../"):
		cleaned = cleaned[3:]
	if cleaned.startswith("./"):
		cleaned = cleaned[2:]
	return REPO_ROOT.joinpath(cleaned)


__all__ = [
	"repo_path",
	"REPO_ROOT",
	"data_path",
	"figs_path",
	"repo_file",
]


def _resolve_path(path: Any) -> Any:
	"""Convert supported path inputs to repository-rooted paths."""
	if isinstance(path, (str, os.PathLike)):
		path_str = str(path)
		if "://" in path_str or path_str.startswith("SCRATCH_DIR"):
			return path
		return repo_file(path_str)
	return path


_PATCHED_IO = False


def _patch_io_helpers() -> None:
	global _PATCHED_IO
	if _PATCHED_IO:
		return

	# Patch builtins.open
	original_open = builtins.open

	def open_repo(path, *args, **kwargs):
		return original_open(_resolve_path(path), *args, **kwargs)

	builtins.open = open_repo

	# Patch pandas.read_csv if available
	if pd is not None:
		original_read_csv = pd.read_csv

		def read_csv_repo(path, *args, **kwargs):
			return original_read_csv(_resolve_path(path), *args, **kwargs)

		pd.read_csv = read_csv_repo

	# Patch torch.load / torch.save if available
	if torch is not None:
		original_torch_load = torch.load
		original_torch_save = torch.save

		def torch_load_repo(f, *args, **kwargs):
			return original_torch_load(_resolve_path(f), *args, **kwargs)

		def torch_save_repo(obj, f, *args, **kwargs):
			return original_torch_save(obj, _resolve_path(f), *args, **kwargs)

		torch.load = torch_load_repo
		torch.save = torch_save_repo

	# Patch numpy load/save helpers commonly used
	if np is not None:
		original_np_load = np.load
		original_np_save = np.save

		def np_load_repo(path, *args, **kwargs):
			return original_np_load(_resolve_path(path), *args, **kwargs)

		def np_save_repo(path, arr, *args, **kwargs):
			return original_np_save(_resolve_path(path), arr, *args, **kwargs)

		np.load = np_load_repo
		np.save = np_save_repo

	# Patch PIL.Image open/save helpers if available
	if Image is not None:
		original_image_open = Image.open
		original_image_save = Image.Image.save

		def image_open_repo(fp, *args, **kwargs):
			return original_image_open(_resolve_path(fp), *args, **kwargs)

		def image_save_repo(self, fp, *args, **kwargs):
			return original_image_save(self, _resolve_path(fp), *args, **kwargs)

		Image.open = image_open_repo
		Image.Image.save = image_save_repo

	_PATCHED_IO = True


_patch_io_helpers()
