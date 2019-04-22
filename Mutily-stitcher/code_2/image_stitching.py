#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

# Built-in Modules
import os
import argparse
import logging



# Built-in Modules
import logging
# Standard Modules
import cv2
import numpy
# Custom Modules

logger = logging.getLogger('main')


def combine_images(img0, img1, h_matrix):
    logger.debug('combining images... ')
    points0 = numpy.array(
        [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]], dtype=numpy.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = numpy.array(
        [[0, 0], [0, img1.shape[0]], [img1.shape[1], img0.shape[0]], [img1.shape[1], 0]], dtype=numpy.float32)
    points1 = points1.reshape((-1, 1, 2))
    points2 = cv2.perspectiveTransform(points1, h_matrix)
    points = numpy.concatenate((points0, points2), axis=0)
    [x_min, y_min] = numpy.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = numpy.int32(points.max(axis=0).ravel() + 0.5)
    H_translation = numpy.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    logger.debug('warping previous image...')
    output_img = cv2.warpPerspective(img1, H_translation.dot(h_matrix), (x_max - x_min, y_max - y_min))
    output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0
    return output_img

	
def compute_matches(features0, features1, matcher, knn=5, lowe=0.7):
    keypoints0, descriptors0 = features0
    keypoints1, descriptors1 = features1

    logger.debug('finding correspondence')

    matches = matcher.knnMatch(descriptors0, descriptors1, k=knn)

    logger.debug("filtering matches with lowe test")

    positive = []
    for match0, match1 in matches:
        if match0.distance < lowe * match1.distance:
            positive.append(match0)

    src_pts = numpy.array([keypoints0[good_match.queryIdx].pt for good_match in positive], dtype=numpy.float32)
    src_pts = src_pts.reshape((-1, 1, 2))
    dst_pts = numpy.array([keypoints1[good_match.trainIdx].pt for good_match in positive], dtype=numpy.float32)
    dst_pts = dst_pts.reshape((-1, 1, 2))

    return src_pts, dst_pts, len(positive)

def is_cv2():
    major, minor, increment = cv2.__version__.split(".")
    return major == "2"


def is_cv3():
    major, minor, increment = cv2.__version__.split(".")
    return major == "3"


def display(title, img, max_size=500000):
    assert isinstance(img, numpy.ndarray), 'img must be a numpy array'
    assert isinstance(title, str), 'title must be a string'
    scale = numpy.sqrt(min(1.0, float(max_size) / (img.shape[0] * img.shape[1])))
    shape = (int(scale * img.shape[1]), int(scale * img.shape[0]))
    img = cv2.resize(img, shape)
    cv2.imshow(title, img)


def save_image(path, result):
    name, ext = os.path.splitext(path)
    img_path = '{0}.png'.format(name)
    logger.debug('writing image to {0}'.format(img_path))
    cv2.imwrite(img_path, result)
    logger.debug('writing complete')
	
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image_paths', type=str, nargs='+', help="paths to one or more images or image directories")
    parser.add_argument('-b', '--debug', dest='debug', action='store_true', help='enable debug logging')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', help='disable all logging')
    parser.add_argument('-d', '--display', dest='display', action='store_true', help="display result")
    parser.add_argument('-s', '--save', dest='save', action='store_true', help="save result to file")
    parser.add_argument("--save_path", dest='save_path', default="stitched.png", type=str, help="path to save result")
    parser.add_argument('-k', '--knn', dest='knn', default=2, type=int, help="Knn cluster value")
    parser.add_argument('-l', '--lowe', dest='lowe', default=0.7, type=float, help='acceptable distance between points')
    parser.add_argument('-m', '--min', dest='min_correspondence', default=10, type=int, help='min correspondences')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main")

    logging.info("beginning sequential matching")

    if is_cv2():
        sift = cv2.SIFT()
    elif is_cv3():
        sift = cv2.xfeatures2d.SIFT_create()
    else:
        raise RuntimeError("error! unknown version of python!")

    result = None
    result_gry = None

    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

    image_paths = args.image_paths
    image_index = -1
    for image_path in image_paths:
        if not os.path.exists(image_path):
            logging.error('{0} is not a valid path'.format(image_path))
            continue
        if os.path.isdir(image_path):
            extensions = [".jpeg", ".jpg", ".png"]
            for file_path in os.listdir(image_path):
                if os.path.splitext(file_path)[1].lower() in extensions:
                    image_paths.append(os.path.join(image_path, file_path))
            continue

        logging.info("reading image from {0}".format(image_path))
        image_colour = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image_colour, cv2.COLOR_RGB2GRAY)

        image_index += 1

        if image_index == 0:
            result = image_colour
            result_gry = image_gray
            continue

        logger.debug('computing sift features')
        features0 = sift.detectAndCompute(result_gry, None)
        features1 = sift.detectAndCompute(image_gray, None)

        matches_src, matches_dst, n_matches = compute_matches(features0, features1, flann, knn=args.knn)

        if n_matches < args.min_correspondence:
            logger.error("error! too few correspondences")
            continue

        logger.debug("computing homography between accumulated and new images")
        H, mask = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5.0)
        print(H)
        result = combine_images(image_colour, result, H)

        if args.display and not args.quiet:
            display('result', result)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        result_gry = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    logger.info("processing complete!")

    if args.display and not args.quiet:
        cv2.destroyAllWindows()
    if args.save:
        logger.info("saving stitched image to {0}".format(args.save_path))
        save_image(args.save_path, result)
