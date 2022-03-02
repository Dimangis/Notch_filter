import pathlib

from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    P, Q = shape
    H = np.zeros((P, Q))
    for u in range(0, P):
        for v in range(0, Q):
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)
            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1.0
    return H

def set_plot_title(title, fs=16):
    plt.title(title, fontsize=fs)


img = cv2.imread("Samples/Bbain_03_55.png", 0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
phase_spectrum = np.angle(fshift)
magnitude_spectrum = 20 * np.log(np.abs(fshift))
cv2.imwrite("Samples/tmp.png", magnitude_spectrum)

img_shape = img.shape

H1 = notch_reject_filter(img_shape, 10, 60, -75)
H2 = notch_reject_filter(img_shape, 10, 40, -50)
H3 = notch_reject_filter(img_shape, 7, 20, -25)
H4 = notch_reject_filter(img_shape, 7, 15, -17.5)
H5 = notch_reject_filter(img_shape, 10, 50, -62.5)
H6 = notch_reject_filter(img_shape, 10, 30, -37.5)

NotchFilter = H1 * H2 * H3 * H4 * H5 * H6
NotchRejectCenter = fshift * NotchFilter
NotchReject = np.fft.ifftshift(NotchRejectCenter)
inverse_NotchReject = np.fft.ifft2(NotchReject)

Result = np.abs(inverse_NotchReject)

plt.imshow(Image.open(pathlib.Path("Samples/tmp.png")), cmap="gray")
set_plot_title("Click on image to choose points. (Press any key to Start)")
plt.waitforbuttonpress()
set_plot_title(f'Select 4 points with mouse click')
points = np.asarray(plt.ginput(6))
plt.close()

print(points)

plt.subplot(222)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(221)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('magnitude spectrum')

plt.subplot(223)
plt.imshow(magnitude_spectrum * NotchFilter, "gray")
plt.title("Notch Reject Filter")

plt.subplot(224)
plt.imshow(Result, "gray")
plt.title("Result")


plt.show()
