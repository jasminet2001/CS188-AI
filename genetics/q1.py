import itertools
from sklearn.datasets import load_iris
import numpy as np
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt

data = load_iris()
X = data.data


def initialize_population(pop_size, num_data_points, max_clusters):
    return np.random.randint(0, max_clusters, size=(pop_size, num_data_points))


def calculate_wcss(chromosome, data_points):
    clusters = {}
    for idx, cluster_id in enumerate(chromosome):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(data_points[idx])

    wcss = 0
    for cluster, points in clusters.items():
        center = np.mean(points, axis=0)
        wcss += np.sum(euclidean_distances(points, [center])**2)
    return wcss


def fitness(chromosome, data_points):
    # Minimize WCSS, hence fitness is inverse
    return 1 / calculate_wcss(chromosome, data_points + 1e-10)


# def fitness(chromosome, data_points):
#     total_distance = 0

#     # Calculate total distance within each cluster
#     for cluster_id in np.unique(chromosome):
#         cluster_points = data_points[chromosome == cluster_id]
#         if len(cluster_points) > 1:
#             cluster_distance = np.sum(euclidean_distances(cluster_points, cluster_points))
#         else:
#             cluster_distance = 0
#         total_distance += cluster_distance

#     # Minimize total distance, hence fitness is inverse
#     return 1 / total_distance if total_distance != 0 else float('inf')


# def crossover(parent1, parent2):
#     crossover_point = np.random.randint(1, len(parent1) - 1)
#     child1 = np.concatenate(
#         [parent1[:crossover_point], parent2[crossover_point:]])
#     child2 = np.concatenate(
#         [parent2[:crossover_point], parent1[crossover_point:]])
#     return child1, child2
def crossover(parent1, parent2):
    # Multi-point crossover
    num_points = np.random.randint(1, min(len(parent1), len(parent2)) - 1)
    crossover_points = np.sort(np.random.choice(
        range(1, min(len(parent1), len(parent2))), size=num_points, replace=False))

    child1, child2 = parent1.copy(), parent2.copy()
    for i in range(0, len(crossover_points), 2):
        start, end = crossover_points[i], crossover_points[i +
                                                           1] if i+1 < len(crossover_points) else len(parent1)
        child1[start:end], child2[start:end] = parent2[start:end], parent1[start:end]
    return child1, child2


# def mutate(chromosome, mutation_rate=0.1):
#     for i in range(len(chromosome)):
#         if np.random.rand() < mutation_rate:
#             chromosome[i] = np.random.randint(0, np.max(chromosome) + 1)
#     return chromosome

def mutate(chromosome, generation, max_generations):
    # Decrease mutation rate over generations
    mutation_rate = 0.1 * (1 - generation / max_generations)
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] = np.random.randint(0, np.max(chromosome) + 1)
    return chromosome


def genetic_algorithm(data_points, pop_size=200, max_clusters=5, num_generations=50, elite_size=5):
    population = initialize_population(
        pop_size, len(data_points), max_clusters)
    best_fitness = float('inf')
    best_solution = None

    for generation in range(num_generations):
        # Evaluate fitness
        fitness_scores = [fitness(chromosome, data_points)
                          for chromosome in population]

        # Sort population based on fitness
        sorted_population = [x for _, x in sorted(
            zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]

        # Best solution in the current population
        current_best = sorted_population[0]
        current_fitness = 1 / fitness(current_best, data_points)

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = current_best

        # Select the elites
        new_population = sorted_population[:elite_size]

        # Create new population using crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection
            parents = np.random.choice(range(pop_size), size=2, replace=False)
            parent1, parent2 = sorted_population[parents[0]
                                                 ], sorted_population[parents[1]]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, generation, num_generations)
            child2 = mutate(child2, generation, num_generations)
            new_population.extend([child1, child2])

        population = new_population[:pop_size]

    return best_solution, best_fitness


best_chromosome, best_score = genetic_algorithm(X)
print("Best Chromosome:", best_chromosome)
print("Best Fitness (Lower is better):", best_score)

best_solution = best_chromosome

# Get all combinations of feature indices
feature_combinations = list(itertools.combinations(range(X.shape[1]), 2))

# Plot each combination of features
plt.figure(figsize=(15, 10))
for i, (feature1_idx, feature2_idx) in enumerate(feature_combinations, 1):
    plt.subplot(2, 3, i)
    for cluster_id in np.unique(best_solution):
        cluster_points = X[best_solution == cluster_id]
        plt.scatter(cluster_points[:, feature1_idx], cluster_points[:,
                    feature2_idx], label=f'Cluster {cluster_id}')
    plt.title(f'Features {feature1_idx+1} vs {feature2_idx+1}')
    plt.xlabel(f'Feature {feature1_idx+1}')
    plt.ylabel(f'Feature {feature2_idx+1}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
