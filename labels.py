"""
bi-directional look ups for emotions wrt numeric labeling
"""

labels_word2num_ = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}


def labels_word2num(word):
    return labels_word2num_[word]


labels_num2word_ = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}


def labels_num2word(num):
    return labels_num2word_[num]
