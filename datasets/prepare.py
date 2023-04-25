import numpy as np
import glob
import os
import imageio.v3 as iio
from tqdm import tqdm
import labels
import h5py

"""
prepare the store data into HDF (H5 file)
"""


def get_emotion_from_path(emotion_path):
    return emotion_path.strip().split("/")[-1]


def prepare(
        path="./fer2013/train_processed/*",
        saveto="./fer2013/train_processed.h5"
):
    pixels = []
    targets = []
    for emotion_path in tqdm(glob.glob(path)):
        emotion = get_emotion_from_path(emotion_path)
        emotion = labels.labels_word2num(emotion)
        for file in glob.glob(emotion_path + "/*.jpg"):
            targets.append(emotion)
            image = iio.imread(file)
            pixels.append(
                image
            )
    pixels = np.array(pixels)
    targets = np.array(targets)

    print(f"-> Finished processing path = {path}, saved to {saveto}.")
    print(
        f"-> Total pixels saved: {pixels.shape}, total labels saved: {targets.shape}")

    hf = h5py.File(saveto, 'w')
    hf.create_dataset("pixels", data=pixels)
    hf.create_dataset("targets", data=targets)
    hf.close()


prepare()
prepare(
    path="./fer2013/test_processed/*",
    saveto="./fer2013/test_processed.h5"
)
