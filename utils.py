import cv2
import numpy as np


def threshold(image, low=0, high=255):
    val_1 = cv2.threshold(image, low, 255, cv2.THRESH_BINARY)[1]
    val_2 = cv2.threshold(image, high, 255, cv2.THRESH_BINARY_INV)[1]
    return cv2.bitwise_and(val_1, val_2)


def ball_separator(image, color):
    match color.lower():
        case "red":
            sep_1 = cv2.threshold(
                image[:, :, 1], 70, 255, cv2.THRESH_BINARY_INV)[1]
            sep_2 = cv2.threshold(
                image[:, :, 2], 80, 255, cv2.THRESH_BINARY)[1]
            return cv2.bitwise_and(sep_1, sep_2)
        case "pink":
            sep_3 = threshold(cv2.cvtColor(
                image, cv2.COLOR_RGB2HSV)[:, :, 1], 120, 180)
            return cv2.morphologyEx(sep_3, cv2.MORPH_OPEN, kernel=np.ones((3, 3)))
        case "blue":
            sep_1 = cv2.threshold(
                image[:, :, 0], 90, 255, cv2.THRESH_BINARY)[1]
            sep_2 = cv2.threshold(
                image[:, :, 2], 23, 255, cv2.THRESH_BINARY_INV)[1]
            return cv2.bitwise_and(sep_1, sep_2)
        case "black":
            sep_1 = cv2.threshold(
                image[:, :, 1], 80, 255, cv2.THRESH_BINARY_INV)[1]
            sep_2 = cv2.threshold(
                image[:, :, 2], 15, 255, cv2.THRESH_BINARY_INV)[1]
            return cv2.bitwise_and(sep_1, sep_2)
        case "green":  # tego sie nie da
            sep = threshold(cv2.cvtColor(
                image, cv2.COLOR_RGB2HSV)[:, :, 0], 44, 57)
            return cv2.morphologyEx(sep, cv2.MORPH_OPEN, kernel=np.ones((3, 3)))
        case "yellow":
            sep_2 = threshold(cv2.cvtColor(
                image, cv2.COLOR_RGB2GRAY), 150, 200)
            sep_1 = cv2.threshold(
                image[:, :, 0], 80, 255, cv2.THRESH_BINARY)[1]
            return np.clip(sep_2.astype(int) - sep_1.astype(int), 0, 255).astype(np.uint8)
        case "white":
            sep_1 = cv2.threshold(
                image[:, :, 2], 230, 255, cv2.THRESH_BINARY)[1]
            sep_2 = cv2.threshold(cv2.cvtColor(
                image, cv2.COLOR_RGB2GRAY), 210, 255, cv2.THRESH_BINARY)[1]
            sep_3 = cv2.threshold(
                image[:, :, 1], 254, 255, cv2.THRESH_BINARY)[1]
            sep_4 = cv2.threshold(
                image[:, :, 0], 170, 255, cv2.THRESH_BINARY)[1]
            return cv2.bitwise_and(cv2.bitwise_and(cv2.bitwise_and(sep_3, sep_1), sep_2), sep_4)
        case "brown":
            sep_1 = threshold(image[:, :, 2], 60, 250)
            sep_2 = cv2.threshold(cv2.cvtColor(
                image, cv2.COLOR_RGB2GRAY), 100, 255, cv2.THRESH_BINARY_INV)[1]
            sep_3 = threshold(image[:, :, 1], 65, 160)
            return cv2.morphologyEx(cv2.bitwise_and(cv2.bitwise_and(sep_1, sep_2), sep_3), cv2.MORPH_OPEN, kernel=np.ones((3, 3)))
        case _:
            raise ValueError
