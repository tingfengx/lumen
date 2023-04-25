import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import glob
import imageio.v3 as iio
import dlib
import os
from tqdm import tqdm

if not os.path.exists("./fer2013"):
    # download_fer2013()
    pass


def train_path(category, filename="*", processed=False):
    if processed:
        return f"./fer2013/train_processed/{category}/{filename}.png"
    return f"./fer2013/train/{category}/{filename}.png"


def test_path(category, filename="*", processed=False):
    if processed:
        return f"./fer2013/test_processed/{category}/{filename}.png"
    return f"./fer2013/test/{category}/{filename}.png"


emotions = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]

if not os.path.exists("./fer2013/test_processed"):
    os.system("mkdir ./fer2013/test_processed")
    for emotion in emotions:
        os.system(f"mkdir ./fer2013/test_processed/{emotion}")
if not os.path.exists("./fer2013/train_processed"):
    os.system("mkdir ./fer2013/train_processed")
    for emotion in emotions:
        os.system(f"mkdir ./fer2013/train_processed/{emotion}")

"""
train set percentages
"""


def calc_percentages():
    detector = dlib.get_frontal_face_detector()
    for emotion in emotions:
        total, withface = 0, 0
        for p in glob.glob(train_path(emotion)):
            im = iio.imread(p)
            bbox = detector(im, 1)
            if bbox:
                withface += 1
            total += 1

        print(emotion, withface, total, f"i.e., {100 * withface / total:.2f}%")


# do preprocess for training data, testing data
fullsize, halfsize, quadsize = 48, 48 // 2, 48 // 4


def do_preprocess():
    detector = dlib.get_frontal_face_detector()
    for pathfunc in [train_path, test_path]:
        for emotion in emotions:
            for p in tqdm(glob.glob(pathfunc(emotion))):
                im = iio.imread(p)
                bboxes = detector(im, 1)
                for bbox in bboxes:
                    top = max(0, bbox.top())
                    bottom = min(bbox.bottom(), fullsize)
                    left = max(0, bbox.left())
                    right = min(bbox.right(), fullsize)

                    new_center = (top + bottom) // 2 - 6
                    if new_center < quadsize:
                        top = 0
                        bottom = halfsize
                    elif new_center > fullsize - quadsize:
                        top = fullsize - quadsize
                        bottom = quadsize
                    else:
                        top = new_center - quadsize
                        bottom = new_center + quadsize
                    im = im[top:bottom, :]

                    save_filename = "pd_" + p.strip().split("/")[-1][:-4]
                    matplotlib.image.imsave(
                        pathfunc(emotion, save_filename, processed=True),
                        im
                    )


do_preprocess()
