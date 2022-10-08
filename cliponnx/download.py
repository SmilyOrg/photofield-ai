from urllib.parse import urlparse
import urllib.request
from tqdm import tqdm
import os

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def ensure_model(path_or_url, dir):
    path = path_or_url
    if path_or_url.startswith("http"):
        u = urlparse(path_or_url)
        filename = u.path.split("/")[-1]
        path = f"{dir}{filename}"
        if not os.path.exists(path):
            download(path_or_url, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return path

def download(url, output_path):
    with DownloadProgressBar(
        unit='B',
        unit_scale=True,
        miniters=1,
        desc=url.split('/')[-1]
    ) as t:
        urllib.request.urlretrieve(
            url,
            filename=output_path,
            reporthook=t.update_to
        )