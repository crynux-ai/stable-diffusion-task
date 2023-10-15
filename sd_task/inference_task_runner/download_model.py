import validators
import hashlib
import os
import urllib3
from sd_task.config import ProxyConfig
from tqdm import tqdm


def check_and_download_model_from_url(model_name: str, cache_dir: str, proxy: ProxyConfig):
    if not validators.url(model_name):
        return model_name

    print("Check and download the model file: " + model_name)

    m = hashlib.sha256()
    m.update(model_name.encode('utf-8'))
    url_hash = m.hexdigest()

    model_folder = os.path.join(cache_dir, url_hash)
    model_file = os.path.join(model_folder, "model.safetensors")

    print("The model file will be saved as: " + model_file)

    # Check if we have already cached the model file
    if os.path.isdir(model_folder):
        if os.path.isfile(model_file):
            print("Found a local cache of the model file. Skip the download")
            return model_file
    else:
        os.mkdir(model_folder, 0o755)

    # Download the model file
    model_file = os.path.join(model_folder, "model.safetensors")

    print("Model file not cached locally. Start the download...")

    try:
        if proxy.host != "":
            proxy_str = proxy.host + ":" + str(proxy.port)

            print("Download using proxy: " + proxy_str)

            default_headers = None
            if proxy.username != "" and proxy.password != "":
                default_headers = urllib3.make_headers(proxy_basic_auth=proxy.username + ':' + proxy.password)
            http = urllib3.ProxyManager(
                proxy_str,
                proxy_headers=default_headers,
                num_pools=1
            )
        else:
            http = urllib3.PoolManager(num_pools=1)

        resp = http.request("GET", model_name, preload_content=False)
        total_bytes = resp.getheader("content-length", None)
        if total_bytes is not None:
            total_bytes = int(total_bytes)

        with tqdm.wrapattr(open(model_file, "wb"), "write",
                           miniters=1, desc=model_file,
                           total=total_bytes) as f_out:
            for chunk in resp.stream(1024):
                if chunk:
                    f_out.write(chunk)
            f_out.flush()

        return model_folder
    except Exception as e:
        # delete the broken file if download failed
        if os.path.isfile(model_file):
            os.remove(model_file)

        raise e
