import numpy as np
import random


def calculate_total_distance(path, distance_matrix):
    return sum(distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1)) + distance_matrix[path[-1], path[0]]


def initialize_population(num_cities, population_size):
    population = [list(np.random.permutation(num_cities))
                  for _ in range(population_size)]
    return population


def evaluate_population(population, distance_matrix):
    fitness_scores = [
        1 / calculate_total_distance(individual, distance_matrix) for individual in population]
    return fitness_scores


def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    selected_indices = np.random.choice(
        len(population), size=2, p=probabilities)
    return population[selected_indices[0]], population[selected_indices[1]]


def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child1 = [None] * size
    child2 = [None] * size
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    def fill_child(child, parent):
        parent_index = 0
        for i in range(size):
            if child[i] is None:
                while parent[parent_index] in child:
                    parent_index += 1
                child[i] = parent[parent_index]
        return child

    child1 = fill_child(child1, parent2)
    child2 = fill_child(child2, parent1)

    return child1, child2


def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]
    return individual


def genetic_algorithm_tsp(distance_matrix, population_size=100, generations=500, mutation_rate=0.01):
    num_cities = distance_matrix.shape[0]
    population = initialize_population(num_cities, population_size)

    for generation in range(generations):
        fitness_scores = evaluate_population(population, distance_matrix)
        new_population = []

        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend(
                [mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

        population = new_population

    fitness_scores = evaluate_population(population, distance_matrix)
    best_index = np.argmax(fitness_scores)
    best_individual = population[best_index]
    best_distance = 1 / fitness_scores[best_index]

    return best_individual, best_distance


distance_matrix = np.array([
    [0, 29, 20, 21, 16],
    [29, 0, 15, 19, 28],
    [20, 15, 0, 13, 25],
    [21, 19, 13, 0, 17],
    [16, 28, 25, 17, 0]
])


best_path, best_distance = genetic_algorithm_tsp(distance_matrix)
print("Best path:", best_path)
print("Best distance:", best_distance)
