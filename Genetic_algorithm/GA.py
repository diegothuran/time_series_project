"""
# Example usage
from genetic import *
target = 371
p_count = 100
i_length = 6
i_min = 0
i_max = 100
p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target),]
for i in xrange(100):
    p = evolve(p, target)
    fitness_history.append(grade(p, target))

for datum in fitness_history:
   print datum
"""
from random import randint, random
from operator import add
import matplotlib.pyplot as plt
from Util import extract_features
import numpy as np
from scipy.spatial import distance

def individual(length, min, max):
    'Create a member of the population.'
    return [ randint(min,max) for x in xrange(length) ]

def population(count, length, min, max):
    """
    Create a number of individuals (i.e. a population).

    count: the number of individuals in the population
    length: the number of values per individual
    min: the minimum possible value in an individual's list of values
    max: the maximum possible value in an individual's list of values

    """
    return [ individual(length, min, max) for x in xrange(count) ]

def fitness(individual, target, frequency=1):
    """
    Determine the fitness of an individual. Higher is better.

    individual: the individual to evaluate
    target: the target number individuals are aiming for
    """
    feature_vector = extract_features(individual, frequency)
    target_features = extract_features(target, frequency)
    distance_ = distance.euclidean(feature_vector, target_features)
    return distance_


def grade(pop, target, frequency=1 ):
    'Find average fitness for a population.'
    summed = reduce(add, (fitness(x, target, frequency) for x in pop))
    return summed / (len(pop) * 1.0)


def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01, frequency=1):
    graded = [(fitness(x, target, frequency), x) for x in pop]
    graded = [x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    # randomly add other individuals to
    # promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)
    # mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)
            # this mutation is not ideal, because it
            # restricts the range of possible values,
            # but the function is unaware of the min/max
            # values used to create the individuals,
            individual[pos_to_mutate] = randint(
                0, max(individual))
    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:half] + female[half:]
            children.append(child)        
    parents.extend(children)
    return parents

def generate_serie(target_serie, lenght, min, max, p_count, iterations, frequency):
    pop = population(p_count, lenght, min, max)
    fitness_history = [grade(pop, target_serie, frequency)]

    for i in xrange(iterations):
        pop = evolve(pop, target_serie, frequency)
        fitness_history.append(grade(pop, target_serie, frequency))

    return pop[0]

def generate_time_serie(target, frequency):

    serie_ = generate_serie(target, len(target), 0, int(max(target)), len(target), 100, frequency=frequency)

    return serie_



if __name__ == '__main__':
    serie = [1991.05, 2306.4, 2604.0, 2992.3, 3722.08, 5226.62, 5989.46, 5614.62,
             5527.0, 5389.8, 5384.4, 3656.2, 4034.8, 4230.0, 4793.2, 5602.0, 5065.0,
             5056.0, 5067.2, 5209.6]

    target = serie
    p_count = 100
    i_lenght = len(serie)
    i_min = 0
    i_max = int(np.array(serie).max()) + 1
    p = population(p_count, i_lenght, i_min, i_max)
    fitness_history = [grade(p, target, 1)]

    for i in xrange(100):
        p = evolve(p, target, frequency=1)
        fitness_history.append(grade(p, target, 1))

    plt.plot(fitness_history)
    plt.show()

    plt.plot(serie)
    plt.plot(p[0])
    plt.show()
