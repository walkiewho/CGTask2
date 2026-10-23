#!/bin/python3
import argparse
import cv2
import numpy as np
from pathlib import Path


class ASCIIConverter:
    CHAR_SET = list(r""" .'`^",:;Il!i><~+_-?][}{1)(|\/tfjrxnuvczXYJCLQ0MW&8#%@""")
    LUMA_COEF = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
    HEIGHT_COEF = 2

    def __init__(self, img_path: Path, width=150, area_interpolation=False):
        self._image_path: Path = self._validate_path(img_path)
        self._default_image: np.ndarray = self._open_image()
        self._changeable_image: np.ndarray = np.copy(self._default_image)
        self._char_width: int = 150
        self.set_width(width)
        self._char_height: int = self._get_height()
        self._char_buffer: np.ndarray | None = None
        self._interpolation: int = cv2.INTER_AREA if area_interpolation else cv2.INTER_LINEAR

    def create_ascii(self) -> None:
        self._changeable_image = np.copy(self._default_image)
        self._proceed_image()
        self._resize_image()
        luma_list = self._to_luma_list()
        self._char_buffer = np.array(ASCIIConverter.CHAR_SET)[luma_list]

    def print_ascii(self) -> None:
        string = self._char_buffer_to_string()
        if string is not None:
            print(string)
        else:
            print('Image is not proceed yet')

    def set_path(self, path: Path):
        self._image_path = self._validate_path(path)
        self._default_image = self._open_image()
        self._char_height = self._get_height()

    @property
    def path(self):
        return self._image_path

    def set_width(self, width: int):
        if width > 0:
            self._char_width = width
            self._char_height = self._get_height()
        else:
            raise ValueError(f"Width must be positive, but got {width}")

    @property
    def width(self) -> int:
        return self._char_width

    @property
    def height(self) -> int:
        return self._char_height

    @property
    def char_buffer(self) -> np.ndarray | None:
        return self._char_buffer

    @property
    def default_image(self) -> np.ndarray:
        return self._default_image

    def _char_buffer_to_string(self) -> str | None:
        if self._char_buffer is not None:
            return '\n'.join(''.join(row) for row in self._char_buffer)
        else:
            return None

    def _proceed_image(self) -> None:
        self._denoise()
        self._sharpen()
        self._srgb_to_linear()

    @staticmethod
    def _validate_path(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError
        return path

    def _open_image(self) -> np.ndarray:
        opened_image = cv2.imread(self._image_path)

        if opened_image is None:
            raise IOError
        else:
            return opened_image

    def _srgb_to_linear(self) -> None:
        normalized_image = self._changeable_image.astype(np.float32) / 255.0
        self._changeable_image = np.where(
            normalized_image <= 0.04045,
            normalized_image / 12.92,
            ((normalized_image + 0.055) / 1.055) ** 2.4
        ).astype(np.float32)

    def _denoise(self) -> None:
        self._changeable_image = cv2.fastNlMeansDenoisingColored(
            self._changeable_image, None,
            h=10, hColor=10,
            templateWindowSize=7, searchWindowSize=21
        )

    def _sharpen(self) -> None:
        blurred = cv2.GaussianBlur(self._changeable_image, (0, 0), 3)
        self._changeable_image = cv2.addWeighted(self._changeable_image, 1.5, blurred, -0.5, 0)

    def _get_height(self) -> int:
        img_height, img_width, _ = self._default_image.shape
        char_height = round(img_height / img_width * self._char_width / ASCIIConverter.HEIGHT_COEF)
        return char_height

    def _resize_image(self) -> None:
        self._changeable_image = cv2.resize(self._changeable_image, (self._char_width, self._char_height), interpolation=self._interpolation)

    def _to_luma_list(self):
        return np.clip(self._changeable_image @ ASCIIConverter.LUMA_COEF * (len(ASCIIConverter.CHAR_SET) - 1), 0, len(ASCIIConverter.CHAR_SET) - 1).astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path("test.jpg"))
    parser.add_argument("--width", type=int, default=150)
    parser.add_argument("--interpolate", action="store_true")
    args = parser.parse_args()

    converter = ASCIIConverter(args.path, width=args.width, area_interpolation=args.interpolate)
    converter.create_ascii()
    converter.print_ascii()