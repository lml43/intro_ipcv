import numpy as np
import cv2
import matplotlib.pylab as plt
from pathlib import Path
from skimage import io


def findLargestContour(contours):
    largest_area = 0
    largest_contour_index = -1
    i = 0
    total_contours = len(contours)
    while i < total_contours:
        area = cv2.contourArea(contours[i])
        if area > largest_area:
            largest_area = area
            largest_contour_index = i
        i += 1

    return largest_area, largest_contour_index


def draw_bounding_box(image, contours):

    largest_area, largest_contour_index = findLargestContour(contours)
    cnt = contours[largest_contour_index]
    x, y, w, h = cv2.boundingRect(cnt)
    padding = 3
    res = image.copy()
    cv2.rectangle(res, (x - padding, y - padding), (x + w + padding, y + h + padding), (0, 255, 0), 2)
    return res


def resolve_image(url, low_green, high_green):
    image = io.imread(url)
    segmented = get_segmented(high_green, image, low_green)
    threshold = get_threshold(segmented)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_result = draw_bounding_box(image, contours)
    return image, threshold, img_result


def get_segmented(high_green, image, low_green):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    curr_mask = cv2.inRange(img_hsv, low_green, high_green)
    img_hsv[curr_mask > 0] = ([60, 255, 255])
    segmented = cv2.bitwise_and(img_hsv, img_hsv, mask=curr_mask)
    return segmented


def get_threshold(segmented):
    img_rgb = cv2.cvtColor(segmented, cv2.COLOR_HSV2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    ret, threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    return threshold


def save_images():
    path_new = str(path.parent.parent.joinpath('res').joinpath(path.name.replace('.png', '_bound.png')))
    cv2.imwrite(path_new, res)
    path_new = str(path.parent.parent.joinpath('res').joinpath(path.name.replace('.png', '_thres.png')))
    cv2.imwrite(path_new, threshold)


def show_images():
    plt.figure()
    f, axarr = plt.subplots(1, 3)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axarr[1].imshow(threshold, cmap='gray')
    axarr[2].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.show()


result = list(Path("leaf_dataset/leaves_testing_set_2/color_images").rglob("*.png"))
variant = 30

low_green = np.array([60 - variant, 70, 10])
high_green = np.array([60 + variant, 255, 250])
for path in result:
    image, threshold, res = resolve_image(path, low_green, high_green)

    save_images()
    # show_images()
