import requests
import os
import zipfile
import tarfile
from urllib.parse import urlparse, unquote

import logging

_logger = logging.getLogger(__name__)


def _detect_file_type_by_magic_number(file_path: str) -> str:
    """
    Detect file type by magic number (file signature)
    Returns file extension (including dot), returns empty string if unknown type
    """
    try:
        with open(file_path, 'rb') as f:
            # Read file header bytes
            header = f.read(8)
            
            if len(header) < 2:
                return ""
            
            # ZIP file: PK\x03\x04
            if header.startswith(b'PK\x03\x04'):
                return ".zip"
            
            # GZIP file: \x1f\x8b
            if header.startswith(b'\x1f\x8b'):
                return ".gz"
            
            # BZIP2 file: BZ
            if header.startswith(b'BZ'):
                return ".bz2"
            
            # XZ file: \xfd7zXZ\x00
            if header.startswith(b'\xfd7zXZ\x00'):
                return ".xz"
            
            # TAR file: ustar (GNU tar) or POSIX tar
            if len(header) >= 262:  # TAR header at offset 257 bytes
                f.seek(257)
                tar_header = f.read(8)
                if tar_header.startswith(b'ustar\x00') or tar_header.startswith(b'ustar '):
                    return ".tar"
            
            # 7Z file: 7z\xbc\xaf\x27\x1c
            if header.startswith(b'7z\xbc\xaf\x27\x1c'):
                return ".7z"
            
            # RAR file: Rar!\x1a\x07
            if header.startswith(b'Rar!\x1a\x07'):
                return ".rar"
            
            return ""
            
    except Exception as e:
        _logger.warning(f"Failed to detect file type by magic number: {e}")
        return ""


def _get_file_extension_from_content_type(content_type: str) -> str:
    """
    Get file extension from Content-Type header
    """
    if not content_type:
        return ""
    
    content_type_lower = content_type.lower()
    
    if "zip" in content_type_lower:
        return ".zip"
    elif "tar" in content_type_lower:
        return ".tar"
    elif "gzip" in content_type_lower or "gz" in content_type_lower:
        return ".gz"
    elif "bzip2" in content_type_lower or "bz2" in content_type_lower:
        return ".bz2"
    elif "xz" in content_type_lower:
        return ".xz"
    elif "7z" in content_type_lower:
        return ".7z"
    elif "rar" in content_type_lower:
        return ".rar"
    
    return ""


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
            if not filename:
                filename = "dataset"

        filename = unquote(filename)
        filename = "".join(c for c in filename if c.isalnum() or c in "._- ")

        file_path = os.path.join(save_dir, filename)
        tmp_file_path = file_path + ".tmp"

        if os.path.exists(file_path):
            _logger.info(f"dataset already exists at {file_path}")
            return _handle_existing_file(file_path, save_dir)

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

        # After download, check file type and add appropriate extension
        if "." not in filename:
            # First try to detect file type by magic number
            detected_ext = _detect_file_type_by_magic_number(tmp_file_path)
            
            if not detected_ext:
                # If magic number detection fails, try Content-Type
                content_type = response.headers.get("Content-Type")
                if content_type:
                    detected_ext = _get_file_extension_from_content_type(content_type)
            
            if detected_ext:
                filename += detected_ext
                file_path = os.path.join(save_dir, filename)
                _logger.info(f"detected file type: {detected_ext}, renamed to {file_path}")

        os.rename(tmp_file_path, file_path)

        _logger.info(f"downloaded dataset to {file_path}")
        return _handle_downloaded_file(file_path, save_dir)

    except Exception as e:
        _logger.error(f"failed to download dataset from {url}: {e}")
        raise e
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def _handle_existing_file(file_path: str, save_dir: str) -> str:
    """Handle existing file - check if it's compressed and extract if needed."""
    return _extract_if_compressed(file_path, save_dir)


def _handle_downloaded_file(file_path: str, save_dir: str) -> str:
    """Handle downloaded file - check if it's compressed and extract if needed."""
    return _extract_if_compressed(file_path, save_dir)


def _extract_if_compressed(file_path: str, save_dir: str) -> str:
    """Extract compressed file if it's a supported format, otherwise return the file path."""
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Handle double extensions like .tar.gz
    if name_without_ext.endswith('.tar'):
        name_without_ext = os.path.splitext(name_without_ext)[0]
    
    extract_dir = os.path.join(save_dir, name_without_ext)
    
    # Check if it's a zip file
    if filename.lower().endswith('.zip'):
        if os.path.exists(extract_dir):
            _logger.info(f"extracted dataset already exists at {extract_dir}")
            return extract_dir
        
        _logger.info(f"extracting zip file {file_path} to {extract_dir}")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        _logger.info(f"extracted dataset to {extract_dir}")
        return extract_dir
    
    # Check if it's a tar file (including .tar.gz, .tar.bz2, etc.)
    elif filename.lower().endswith(('.tar', '.tar.gz', '.tar.bz2', '.tar.xz')):
        if os.path.exists(extract_dir):
            _logger.info(f"extracted dataset already exists at {extract_dir}")
            return extract_dir
        
        _logger.info(f"extracting tar file {file_path} to {extract_dir}")
        mode = 'r'
        if filename.lower().endswith('.gz'):
            mode = 'r:gz'
        elif filename.lower().endswith('.bz2'):
            mode = 'r:bz2'
        elif filename.lower().endswith('.xz'):
            mode = 'r:xz'
        
        with tarfile.open(file_path, mode) as tar_ref:
            tar_ref.extractall(extract_dir)
        _logger.info(f"extracted dataset to {extract_dir}")
        return extract_dir
    
    # If filename has no extension, try to detect file type by magic number
    elif "." not in filename:
        detected_ext = _detect_file_type_by_magic_number(file_path)
        if detected_ext:
            _logger.info(f"detected file type by magic number: {detected_ext}")
            # Recursively call self to handle the detected file type
            new_file_path = file_path + detected_ext
            os.rename(file_path, new_file_path)
            return _extract_if_compressed(new_file_path, save_dir)
    
    # Not a compressed file, return the original file path
    return file_path
