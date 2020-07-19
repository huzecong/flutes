import functools
import os
import tarfile
import tempfile
import urllib.request
import zipfile
from typing import Optional

from . import remove_suffix
from .log import log
from .types import BarFn, PathType

__all__ = [
    "download",
]


def download(url: str, save_dir: Optional[PathType] = None, filename: Optional[str] = None, extract: bool = False,
             progress: bool = False, bar_fn: Optional[BarFn] = None, **kwargs) -> str:
    r"""Download a file from the given URL. If the given file already exists in the save directory, download is skipped.
    Supported URL types include:

    - Any direct URL to files.
    - Google Drive shared file URLs in the form of ``https://drive.google.com/file/d/<file_id>/view``.

    :param url: The URL from which to download.
    :param save_dir: The directory to save the file. The directory is created if it doesn't exist. If ``None``, a
        temporary directory is used, and the user is responsible for removing the file.
    :param filename: The name of the downloaded file. If ``None``, the default filename from the URL is used.
    :param extract: Whether to extract compressed files. Defaults to ``False``.
    :param progress: Whether to show download progress as a progress bar. Defaults to ``False``. Note that files
        from Google Drive does not contain file size estimates during downloading.
    :param bar_fn: An optional callable that constructs a progress bar when called. This is useful when you want to
        override the default progress bar, for instance, to use with :class:`~flutes.ProgressBarManager`:

        .. code:: python

            def process(path: str, bar: flutes.ProgressBarManager.Proxy):
                with flutes.progress_open(path, bar_fn=bar.new) as f:
                    ...

    :param kwargs: Additional arguments to pass to `tqdm <https://tqdm.github.io/>`_ initializer.
    :returns: Path to the downloaded file.
    """
    if save_dir is None:
        save_dir_str = tempfile.gettempdir()
    else:
        save_dir_str = str(save_dir)
        os.makedirs(save_dir_str, exist_ok=True)

    if filename is None:
        if 'drive.google.com' in url:
            filename = _extract_google_drive_file_id(url)
        else:
            filename = url.split('/')[-1]
            # If downloading from GitHub, remove suffix ?raw=true from local filename.
            filename = remove_suffix(filename, "?raw=true")

    if progress:
        if bar_fn is None:
            from tqdm import tqdm
            bar_fn = tqdm
        bar_fn = functools.partial(bar_fn, **kwargs)
    else:
        bar_fn = None

    filepath = os.path.join(save_dir_str, filename)
    if not os.path.exists(filepath):
        if 'drive.google.com' in url:
            filepath = _download_from_google_drive(url, filename, save_dir_str, bar_fn)
        else:
            filepath = _download(url, filename, save_dir_str, bar_fn)

        if extract:
            if tarfile.is_tarfile(filepath):
                with tarfile.open(filepath, 'r') as tfile:
                    tfile.extractall(save_dir_str)
            elif zipfile.is_zipfile(filepath):
                with zipfile.ZipFile(filepath) as zfile:
                    zfile.extractall(save_dir_str)
            else:
                log("Unknown compression type. Only .tar.gz, .tar.bz2, .tar, and .zip are supported", "warning")

    return filepath


def _download(url: str, filename: str, path: str, bar_fn: Optional[BarFn] = None) -> str:
    if bar_fn is None:
        progress = _progress_hook = None
    else:
        progress = None
        prev_count = 0

        def _progress_hook(count, block_size, total_size):
            nonlocal progress, prev_count
            if progress is None:
                progress = bar_fn()
            if total_size != -1 and progress.total is None:
                progress.total = total_size
                progress.refresh()
            if count > prev_count:
                progress.update((count - prev_count) * block_size)
                prev_count = count

    filepath = os.path.join(path, filename)
    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress_hook)
    if progress is not None:
        progress.close()
    return filepath


def _extract_google_drive_file_id(url: str) -> str:
    # The ID is the first segment after `/d/`.
    url_suffix = url[url.find('/d/') + 3:]
    return url_suffix.split('/')[0]


def _download_from_google_drive(url: str, filename: str, path: str, bar_fn: Optional[BarFn] = None) -> str:
    # Credit: https://github.com/saurabhshri/gdrive-downloader
    import requests

    def _get_confirm_token(resp):
        for key, value in resp.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    file_id = _extract_google_drive_file_id(url)

    gurl = "https://docs.google.com/uc?export=download"
    sess = requests.Session()
    response = sess.get(gurl, params={'id': file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = sess.get(gurl, params=params, stream=True)

    filepath = os.path.join(path, filename)
    CHUNK_SIZE = 32768
    progress = bar_fn() if bar_fn is not None else None
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                if progress is not None:
                    progress.update(len(chunk))
                f.write(chunk)
    if progress is not None:
        progress.close()
    return filepath
