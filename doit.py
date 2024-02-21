import collections
import random
import re

import cv2
import imagehash
import numpy as np
import PIL


MTG_CARD_RATIO = 1050 / 750


def apply_clahe(image):
    # Apply CLAHE to the Light channel in LAB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    image[:,:,0] = clahe.apply(image[:,:,0])
    return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)


def color_generator():
    while True:
        yield (
            random.randint(96, 255),
            random.randint(96, 255),
            random.randint(96, 255),
        )


def extract_rectangle(image, rectangle):
    # Get the rotated rectangle vertices
    box = cv2.boxPoints(rectangle)

    # Compute the upright rectangle
    width = int(rectangle[1][0])
    height = int(rectangle[1][1])
    src_pts = np.float32([box[0], box[1], box[2]])
    dst_pts = np.float32([[0, height-1], [0, 0], [width-1, 0]])
    M = cv2.getAffineTransform(src_pts, dst_pts)

    # Apply the affine transformation
    extracted_rect = cv2.warpAffine(image, M, (width, height))

    # Just assume that a tilted rectangle needs to turn clockwise
    # TODO: Make more flexible?
    if extracted_rect.shape[0] < extracted_rect.shape[1]:
        extracted_rect = cv2.rotate(extracted_rect, cv2.ROTATE_90_CLOCKWISE)

    shrink_percent = 1
    shrink = int(extracted_rect.shape[0] * shrink_percent/100)
    return extracted_rect[shrink:-shrink, shrink:-shrink]


def get_ratio(rectangle):
    # Get long side vs short side ratio of a rectangle (more than 1 in a rectangle)
    size = rectangle[1]

    return max(size) / min(size)


def get_rectangle_size(rectangle):
    # Expecting a rectangle: ((x, y), (w, h), angle)

    return rectangle[1][0] * rectangle[1][1]


def get_top_left(rectangle):
    # Get the coordinate/point of the top left corner of rectangle
    box = np.intp(cv2.boxPoints(rectangle))
    return box[np.min(np.sum(box, axis=1)) == np.sum(box, axis=1)][0]


def is_rect_within(rectangle1, rectangle2):
    box = np.intp(cv2.boxPoints(rectangle1))
    return cv2.pointPolygonTest(box, rectangle2[0], False) > 0  # positive (inside), negative (outside), or zero (on an edge)


def is_straight(rectangle, tolerance=5):
    # Expecting a rectangle: ((x, y), (w, h), angle)

    angle = rectangle[2]
    return any(np.isclose([0, 90], angle, atol=tolerance))


def load_hashes(filepath):
    hashes = {}
    with open(filepath) as f:
        for line in f:
            hash_str, name = line.strip().split(" ", 1)
            _hash = imagehash.hex_to_hash(hash_str)
            hashes[_hash] = name  # TODO: Handle if allready set?
    return hashes


print("Loading hashes...")
hashes = load_hashes("hashes.txt")
print("Done loading hashes")

random_colors = color_generator()

original_img = cv2.imread("input.png")
#original_img = cv2.rotate(original_img, cv2.ROTATE_180)

img = original_img.copy()
img = apply_clahe(img)

img2 = img.copy()

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#img2 = cv2.equalizeHist(img2)
img2 = cv2.GaussianBlur(img2, (3, 3), 0)
#img2 = cv2.normalize(img2,  np.zeros((800,800)), 0, 255, cv2.NORM_MINMAX)

thresh = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 5)

#cv2.imwrite("out_img2.png", img2)
#cv2.imwrite("out_thresh.png", thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# TODO: Process via hierarchy?

rectangles = [cv2.minAreaRect(contour) for contour in contours]
angles = [rectangle[2] for rectangle in rectangles]
rectangle_sizes = [get_rectangle_size(rectangle) for rectangle in rectangles]
biggest_rectangle_size = max(rectangle_sizes)

skip_counter = collections.Counter()

card_counter = 0
for index, rectangle in enumerate(rectangles):
    center, size, angle = rectangle
    rectangle_size = get_rectangle_size(rectangle)

    # Skip rectangles with zero width or height
    if size[0] == 0 or size[1] == 0:
        skip_counter.update(["zero-edge"])
        continue

    # Skip all rectangles that are small
    # Hard value => very dependant on image size
    if rectangle_size < 1000:
        skip_counter.update(["small"])
        continue

    # Skip all rectangles that aren't big ish
    # (It's assumed that the biggest rectangle found is a MTG card)
    if rectangle_size < biggest_rectangle_size * 0.8:
        skip_counter.update(["size"])
        continue

    # Skip all rectangles at an significant angle
    #if not is_straight(rectangle):
    #    skip_counter.update(["angle"])
    #    continue

    # Skip all rectangles that are way of the ratios of a MTG card
    if not np.isclose(MTG_CARD_RATIO, get_ratio(rectangle), rtol=0.1):
        skip_counter.update(["ratio"])
        continue

    # Skip rectangles that are inside other rectangles
    inside_other = False
    for rectangle2 in rectangles:
        if rectangle == rectangle2:
            continue
        if rectangle_size < get_rectangle_size(rectangle2) and is_rect_within(rectangle2, rectangle):
            inside_other = True
            break
    if inside_other:
        skip_counter.update(["subset"])
        continue

    card_counter += 1

    box = np.intp(cv2.boxPoints(rectangle))

    color = next(random_colors)
    cv2.drawContours(img, [box], 0, color, 5)

    cv2.putText(img, f"{card_counter} {str(round(angle, 1))}", (int(center[0]) - 50, int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

    cv2.circle(img, np.intp(box[0]), 10, (0, 0, 64), 5)
    top_left = get_top_left(rectangle)
    cv2.circle(img, np.intp(top_left), 15, (0, 0, 255), 5)

    card = extract_rectangle(original_img, rectangle)
    card_rgb = card[..., ::-1]  # Unfuck BGR RGB
    card_image = PIL.Image.fromarray(card_rgb)
    cv2.imwrite(f"out_card_{card_counter}.png", card)
    phash = imagehash.phash(card_image)
    best_match = 1000  # Very high value compared to real comparison
    name = None
    for _hash in hashes.keys():
        if (match := _hash - phash) < best_match:
            best_match = match
            name = hashes[_hash]
    print(card_counter, phash, best_match, name)
    cv2.putText(img, name, np.intp(top_left), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 8)
    cv2.putText(img, name, np.intp(top_left), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imwrite("output.png", img)
