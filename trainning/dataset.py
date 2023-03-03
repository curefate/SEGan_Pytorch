import os
import numpy as np
import zipfile
from PIL import Image
from torch.utils import data

try:
    import pyspng
except ImportError:
    pyspng = None


# TODO DATASET LABEL
def get_label(idx):
    return idx


def get_file_ext(fname):
    return os.path.splitext(fname)[1].lower()


class DatasetReader(data.Dataset):
    def __init__(self,
                 path,
                 transform=None,
                 resolution=None
                 ):
        self._path = path
        self._transform = transform
        self._zipfile = None

        # --------------------------------------
        # 获取所有图片的文件名到_all_fnames
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in
                                os.walk(self._path) for fname in files}
        elif get_file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        Image.init()
        # --------------------------------------
        # 对所有的图片file进行排序,到_image_fnames
        self._image_fnames = sorted(fname for fname in self._all_fnames if get_file_ext(fname) in Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        # --------------------------------------
        # 得到shape(pic_nums, h, w, c) 和 idx
        self._raw_shape = list([len(self._image_fnames)] + list(np.array(self._load_raw_image(0)).shape))
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)

        # --------------------------------------
        # 检查resolution
        if resolution is not None and (self._raw_shape[1] != resolution or self._raw_shape[2] != resolution):
            raise IOError('Image files do not match the specified resolution')

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        if self._transform is not None:
            image = self._transform(image.copy())
        else:
            image = image.copy()
        return image, get_label(idx)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and get_file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = Image.open(f).copy()
        return image

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile
