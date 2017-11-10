from sklearn.tree import DecisionTreeClassifier
from Genetic_algorithm import GA
import numpy as np
import matplotlib.pyplot as plt

def train_classifier(cls, input_data, label_data):

    cls.fit(input_data, label_data)

    return cls

def generate_time_serie(target, frequency):
    serie_ = GA.generate_serie(target, len(target), 0, int(np.array(target).max()), len(target), 100, frequency=frequency)

    return serie_


if __name__ == '__main__':
    serie = [1991.05, 2306.4, 2604.0, 2992.3, 3722.08, 5226.62, 5989.46, 5614.62,
             5527.0, 5389.8, 5384.4, 3656.2, 4034.8, 4230.0, 4793.2, 5602.0, 5065.0,
             5056.0, 5067.2, 5209.6]

    serie_ = generate_time_serie(serie, 1)

    plt.plot(serie)
    plt.plot(serie_)
    plt.show()