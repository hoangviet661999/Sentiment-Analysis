import numpy as np
import collections

def read_data(text_path, labels_path):

    with open(text_path) as f:
        messages = [message for message in f.readlines()]
    f.close()

    with open(labels_path) as f:
        labels = [label for label in f.readlines()]
    f.close()

    return messages, labels

def get_word(message):
    return message.lower().split()

def create_dictionary(messages):
    words = [word for message in messages for word in get_word(message)]
    word_count = collections.Counter(words)
    freq_words = [word for word, count in word_count.items() if count>=5]
    return { word: index for index, word in enumerate(freq_words) }

def transform_text(messages, word_dictionary):

    m, n = len(messages), len(word_dictionary)
    word_counts = [collections.Counter(get_word(message)) for message in messages]

    matrix = np.zeros((m, n), dtype=int)

    for i in range(m):
        for word, count in word_counts[i].items():
            if word in word_dictionary:
                matrix[i][word_dictionary[word]] += count

    return matrix

