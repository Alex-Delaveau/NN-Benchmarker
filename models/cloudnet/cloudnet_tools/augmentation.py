import numpy as np
from skimage import transform

"""
Some lines borrowed from: https://www.kaggle.com/sashakorekov/end-to-end-resnet50-with-tta-lb-0-93
"""

def rotate_clk_img_and_msk(img: np.ndarray, msk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = np.random.choice([4, 6, 8, 10, 12, 14, 16, 18, 20])
    img_o = transform.rotate(img, angle, resize=False, preserve_range=True, mode='symmetric')
    msk_o = transform.rotate(msk, angle, resize=False, preserve_range=True, mode='symmetric')
    return img_o, msk_o

def rotate_cclk_img_and_msk(img: np.ndarray, msk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = np.random.choice([-20, -18, -16, -14, -12, -10, -8, -6, -4])
    img_o = transform.rotate(img, angle, resize=False, preserve_range=True, mode='symmetric')
    msk_o = transform.rotate(msk, angle, resize=False, preserve_range=True, mode='symmetric')
    return img_o, msk_o

def flipping_img_and_msk(img: np.ndarray, msk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img_o = np.flip(img, axis=1)
    msk_o = np.flip(msk, axis=1)
    return img_o, msk_o

def zoom_img_and_msk(img: np.ndarray, msk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zoom_factor = np.random.choice([1.2, 1.5, 1.8, 2, 2.2, 2.5])
    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    img_zoomed = transform.resize(img, (zh, zw), preserve_range=True, mode='symmetric')
    msk_zoomed = transform.resize(msk, (zh, zw), preserve_range=True, mode='symmetric')
    region = np.random.choice([0, 1, 2, 3, 4])

    # zooming out
    if zoom_factor <= 1:
        outimg = img_zoomed
        outmsk = msk_zoomed
    else:
        # bounding box of the clipped region within the input array
        if region == 0:
            outimg = img_zoomed[0:h, 0:w]
            outmsk = msk_zoomed[0:h, 0:w]
        elif region == 1:
            outimg = img_zoomed[0:h, zw - w:zw]
            outmsk = msk_zoomed[0:h, zw - w:zw]
        elif region == 2:
            outimg = img_zoomed[zh - h:zh, 0:w]
            outmsk = msk_zoomed[zh - h:zh, 0:w]
        elif region == 3:
            outimg = img_zoomed[zh - h:zh, zw - w:zw]
            outmsk = msk_zoomed[zh - h:zh, zw - w:zw]
        elif region == 4:
            marh = h // 2
            marw = w // 2
            outimg = img_zoomed[(zh // 2 - marh):(zh // 2 + marh), (zw // 2 - marw):(zw // 2 + marw)]
            outmsk = msk_zoomed[(zh // 2 - marh):(zh // 2 + marh), (zw // 2 - marw):(zw // 2 + marw)]

    # to make sure the output is in the same size of the input
    img_o = transform.resize(outimg, (h, w), preserve_range=True, mode='symmetric')
    msk_o = transform.resize(outmsk, (h, w), preserve_range=True, mode='symmetric')
    return img_o, msk_o