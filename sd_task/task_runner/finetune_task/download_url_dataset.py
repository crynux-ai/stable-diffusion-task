import requests
import os
from urllib.parse import urlparse, unquote

import logging

_logger = logging.getLogger(__name__)


def download_dataset_from_url(url: str, save_dir: str = ".") -> str:
    tmp_file_path: str | None = None
    try:
        response = requests.head(url, allow_redirects=True)
        response.raise_for_status()

        filename: str | None = None
        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition:
            import re

            filename_match = re.search(
                r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disposition
            )
            if filename_match:
                filename = filename_match.group(1).strip("\"'")

        if not filename:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or "." not in filename:
                filename = "dataset"

        filename = unquote(filename)

        filename = "".join(c for c in filename if c.isalnum() or c in "._- ")

        file_path = os.path.join(save_dir, filename)
        tmp_file_path = file_path + ".tmp"

        if os.path.exists(file_path):
            _logger.info(f"dataset already exists at {file_path}")
            return file_path

        _logger.info(f"start downloading dataset to {file_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        downloaded = 0
        with open(tmp_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        _logger.info(
                            f"downloading dataset to {file_path} {downloaded / 1024} kb {progress:.1f}%"
                        )
                    else:
                        _logger.info(
                            f"downloading dataset to {file_path} {downloaded / 1024} kb"
                        )

        os.rename(tmp_file_path, file_path)

        _logger.info(f"downloaded dataset to {file_path}")
        return file_path

    except Exception as e:
        _logger.error(f"failed to download dataset from {url}: {e}")
        raise e
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
