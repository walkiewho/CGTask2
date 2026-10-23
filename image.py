import cv2
import numpy as np
from pathlib import Path

CHAR_WIDTH = 200
HEIGHT_COEF = 0.5
CHAR_SET = list(' .\'`^",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYJCLQ0MW&8#%@')


def open_image(path: Path):
    if not path.exists():
        raise FileNotFoundError

    img = cv2.imread(path)

    if img is None:
        raise IOError
    else:
        return img


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    normalized_image = img.astype(np.float32) / 255.0
    linear = np.where(
        normalized_image <= 0.04045,
        normalized_image / 12.92,
        ((normalized_image + 0.055) / 1.055) ** 2.4
    )
    return linear.astype(np.float32)

def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(
        img, None,
        h=10, hColor=10,
        templateWindowSize=7, searchWindowSize=21
    )

def sharpen(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharp

def proceed_image(img: np.ndarray) -> np.ndarray:
    denoised_image = denoise(img)
    sharpen_denoised_image = sharpen(denoised_image)
    result_img = srgb_to_linear(sharpen_denoised_image)

    return result_img

def get_ascii_height(img):
    img_height, img_width, _ = img.shape
    char_height = round(img_height / img_width * CHAR_WIDTH * HEIGHT_COEF)
    return char_height

def resize_image(img):
    new_height = get_ascii_height(img)
    resized = cv2.resize(img, (CHAR_WIDTH, new_height), interpolation=cv2.INTER_AREA)
    return resized

def to_luma_list(img):
    coefs = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
    return img @ coefs



img = open_image(Path("test.jpg"))
new_img = proceed_image(img)
resized_image = resize_image(new_img)
cv2.imwrite('output2.jpg', resized_image * 255)
luma_list = np.clip(to_luma_list(resized_image) * (len(CHAR_SET) - 1), 0, len(CHAR_SET) - 1).astype(np.uint8)
char_buffer = np.array(CHAR_SET)[luma_list]
header = footer = '=' * CHAR_WIDTH
result = header + '\n' + '\n'.join(''.join(row) for row in char_buffer) + '\n' + footer
print(result)