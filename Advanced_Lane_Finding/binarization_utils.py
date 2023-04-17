import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


# selected threshold to highlight yellow lines
# H- Hue - The type of Color itself
# S - Saturation - Intensity of Color
# V - Value of Brightness
yellow_HSV_th_min = np.array([0, 70, 70])
yellow_HSV_th_max = np.array([50, 255, 255])


def thresh_frame_in_HSV(frame, min_values, max_values, verbose=False):
    """
    Threshold a color frame in HSV space
    """
    HSV = cv2.cvtColor(
        frame, cv2.COLOR_BGR2HSV)  # converts the input color frame from the BGR color space (used by OpenCV) to the HSV color space using the cv2.cvtColor() function.

    # axis=2 is for Value, np.all() gives boolean values
    min_th_ok = np.all(HSV > min_values, axis=2)
    max_th_ok = np.all(HSV < max_values, axis=2)

    # np.logical_and() function, which returns a boolean mask where each pixel is True if and only if the corresponding pixel in both input masks is also True
    out = np.logical_and(min_th_ok, max_th_ok)

    if verbose:
        plt.imshow(out, cmap='gray')
        plt.show()

    return out


def thresh_frame_sobel(frame, kernel_size):
    """
    Apply Sobel edge detection to an input frame, then threshold the result
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #  The result of each Sobel operation is the gradient of the image in the respective direction, which indicates the rate of change of pixel intensity in that direction.

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Next, the magnitude of the gradient is computed as the square root of the sum of the squares of the x and y gradients, and is stored in the variable sobel_mag. This magnitude represents the strength of the edge at each pixel, which is high where the rate of change of intensity in both the x and y directions is high
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Normaalize from range of 0 - 255 due to the image 8bit color depth images
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

    # The Sobel operator works by convolving the image with the two kernels to obtain two gradients: one representing changes in pixel intensities in the x-direction (horizontal gradient) and the other in the y-direction (vertical gradient).
    # The magnitude of the gradient at each pixel is then calculated using the Pythagorean theorem, which is used to determine the strength of the edge at that pixel. This edge strength or magnitude is used to create an edge map or edge image

    _, sobel_mag = cv2.threshold(sobel_mag, 50, 1, cv2.THRESH_BINARY)
    # 50: the threshold value. Any pixel value in sobel_mag that is less than 50 is set to 0, and any pixel value greater than or equal to 50 is set to 1.
    # 1: the maximum value to use for the thresholded image. In this case, we set it to 1 to create a binary image with values of either 0 or 1.
    # cv2.THRESH_BINARY: the thresholding method. This method sets all pixel values below the threshold to 0 and all pixel values above or equal to the threshold to the maximum value specified (which is 1 in this case).

    return sobel_mag.astype(bool)


def get_binary_from_equalized_grayscale(frame):
    """
    Apply histogram equalization to an input frame, threshold it and return the (binary) result.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eq_global = cv2.equalizeHist(gray)

    _, th = cv2.threshold(eq_global, thresh=250,
                          maxval=255, type=cv2.THRESH_BINARY)

    return th


def binarize(img, verbose=False):
    """
    Convert an input frame to a binary image which highlight as most as possible the lane-lines.

    :param img: input color frame
    :param verbose: if True, show intermediate results
    :return: binarized frame
    """
    h, w = img.shape[:2]

    binary = np.zeros(shape=(h, w), dtype=np.uint8)

    # highlight yellow lines by threshold in HSV color space
    HSV_yellow_mask = thresh_frame_in_HSV(
        img, yellow_HSV_th_min, yellow_HSV_th_max, verbose=False)
    binary = np.logical_or(binary, HSV_yellow_mask)

    # highlight white lines by thresholding the equalized frame
    eq_white_mask = get_binary_from_equalized_grayscale(img)
    binary = np.logical_or(binary, eq_white_mask)

    # get Sobel binary mask (thresholded gradients)
    sobel_mask = thresh_frame_sobel(img, kernel_size=9)
    binary = np.logical_or(binary, sobel_mask)

    # apply a light morphology to "fill the gaps" in the binary image
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary.astype(
        np.uint8), cv2.MORPH_CLOSE, kernel)

    if verbose:
        f, ax = plt.subplots(2, 3)
        f.set_facecolor('white')
        ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title('input_frame')
        ax[0, 0].set_axis_off()
        ax[0, 0].set_axis_bgcolor('red')
        ax[0, 1].imshow(eq_white_mask, cmap='gray')
        ax[0, 1].set_title('white mask')
        ax[0, 1].set_axis_off()

        ax[0, 2].imshow(HSV_yellow_mask, cmap='gray')
        ax[0, 2].set_title('yellow mask')
        ax[0, 2].set_axis_off()

        ax[1, 0].imshow(sobel_mask, cmap='gray')
        ax[1, 0].set_title('sobel mask')
        ax[1, 0].set_axis_off()

        ax[1, 1].imshow(binary, cmap='gray')
        ax[1, 1].set_title('before closure')
        ax[1, 1].set_axis_off()

        ax[1, 2].imshow(closing, cmap='gray')
        ax[1, 2].set_title('after closure')
        ax[1, 2].set_axis_off()
        plt.show()

    return closing


if __name__ == '__main__':

    test_images = glob.glob('test_images/*.jpg')
    for test_image in test_images:
        img = cv2.imread(test_image)
        binarize(img=img, verbose=True)
