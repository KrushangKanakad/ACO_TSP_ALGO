import numpy as np
import random
from multiprocessing import Pool

'''
procedure ACO_MetaHeuristic is
    while not terminated do
        generateSolutions()
        daemonActions()
        pheromoneUpdate()
    repeat
end procedure
'''

class Graph():
    def _init_(self, nodes, distance, default_pheromone_level = None):
        self.nodes = nodes
        self.distance = distance
        assert distance.shape[1] == distance.shape[0]
        if default_pheromone_level:
            self.intensity = np.full_like(distance, default_pheromone_level).astype('float64')
        else:
            self.intensity = np.full_like(distance, self.distance.mean()*10).astype('float64')
        

    def _str_(self):
        return f'nodes: {str(self.nodes)}\n{self.distance}\n{self.intensity}'

'''
The general algorithm is relatively simple and based on a set of ants, 
each making one of the possible round-trips along the cities. 
At each stage, the ant chooses to move from one city to another according to some rules:

- It must visit each city exactly once;
- A distant city has less chance of being chosen (the visibility);
- The more intense the pheromone trail laid out on an edge between two cities, the greater the probability that that edge will be chosen;
- Having completed its journey, the ant deposits more pheromones on all edges it traversed, if the journey is short;
- After each iteration, trails of pheromones evaporate.

Random ish complete graph: Graph(4, [[0,10,15,20],[10,0,35,25],[10,35,0,30],[20,25,30,0]])
'''

test_graph = Graph(4, np.asarray([[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]]).astype('float64'), )

best_so_far = np.zeros(200, dtype=np.float64)

def cycle_length(g, cycle):
    length = 0
    i = 0
    while i < len(cycle) -1:
        length += g.distance[cycle[i]][cycle[i+1]]
        i+=1
    length+= g.distance[cycle[i]][cycle[0]]
    return length

def add_artificial_good_cycle(g):
    size = g.distance.shape[0]

    for i in range(size-1):
        g.distance[i][i+1]/=10
    g.distance[size-1][0]/=10





def ant_colony_optimization(g, verbose=True, iterations = 100, ants_per_iteration = 50, q = None, degradation_factor = .9, use_inertia = False, run_experiment_break=False, run_experiment_artificial_good_cycle=False):
    total_ants = 0
    
    if q is None:
        q = g.distance.mean()

    best_cycle = None #best_so_far #hardcoded instance. 
    best_length = float('inf') #cycle_length(g, best_so_far) #hardcoded instance. Else use inf

    old_best = None
    inertia = 0
    patience = 100
    index = None
    if run_experiment_break or run_experiment_artificial_good_cycle:
        pheromone_history = []

    for iteration in range(iterations):
        # print(f'iteration {iteration} \n' if (verbose and iteration%1==0) else '', end='')
        # print(f'best weight so far: {round(best_length,2)}\n' if (verbose and iteration%1==0) else '', end='')
        best_so_far[iteration]=round(best_length,2)
        # print(f'average intensity {g.intensity.mean()}\n' if (verbose and iteration%1==0) else '', end='')

        if iteration == 500:
            if run_experiment_artificial_good_cycle:
                add_artificial_good_cycle(g)
            if run_experiment_break:
                index = break_most_traversed_edge(g, 10)
        if iteration >= 500:
            if add_artificial_good_cycle:
                levels = []
                size = g.distance.shape[0]
                for i in range(size-1):
                    levels.append(g.intensity[i][i+1])
                levels.append(g.intensity[size-1][0])
                pheromone_history.append(levels)

            if run_experiment_break:
                pheromone_history.append(g.intensity[index])


        cycles = [traverse_graph(g, random.randint(0, g.nodes -1)) for _ in range(ants_per_iteration)]

        cycles.sort(key = lambda x: x[1])
        cycles = cycles[: ants_per_iteration//2]
        total_ants+=ants_per_iteration

        if best_cycle: #elitism
            cycles.append((best_cycle, best_length))
            
            if use_inertia:
                old_best = best_length

        for cycle, total_length in cycles:

            total_length = cycle_length(g, cycle)
            if total_length < best_length:
                best_length = total_length
                best_cycle = cycle

            delta = q/total_length
            i = 0
            while i < len(cycle) -1:
                g.intensity[cycle[i]][cycle[i+1]]+= delta
                i+=1
            g.intensity[cycle[i]][cycle[0]] += delta
            g.intensity *= degradation_factor
        
        
        if use_inertia and best_cycle:
                        
            if old_best == best_length:
                    inertia+=1
            else:
                inertia = 0

            if inertia > patience:
                print('applying shake')
                g.intensity += g.intensity.mean()
        
    if run_experiment_break or run_experiment_artificial_good_cycle:
        with open('phero_history_exp2_10.txt','w') as f:
            f.write(str(pheromone_history))

    return best_cycle

'''
     -
    - -
'''

def traverse_graph(g, source_node = 0):
    visited = np.asarray([1 for _ in range(g.nodes)])
    visited[source_node] = 0

    cycle = [source_node]
    steps = 0
    current = source_node
    total_length = 0
    while steps < g.nodes -1:

        jumps_neighbors = []
        jumps_values = []
        for node in range(g.nodes):
            if visited[node] != 0:
               sediment = max(g.intensity[current][node], 1e-5)
               v = (sediment*0.9 ) / (g.distance[current][node]*1.5) 
               jumps_neighbors.append(node)
               jumps_values.append(v)

        #jumps = (g.intensity[current]0.9 ) / ((g.distance[current]+0.00001)*1.5)
        #jumps = np.where(visited > 1e-5, jumps, 0.)
        next_node = random.choices(jumps_neighbors, weights = jumps_values)[0]
        
        visited[next_node] = 0
        
        current = next_node
        cycle.append(current)
        steps+=1

    total_length = cycle_length(g, cycle)
    assert len(list(set(cycle))) == len(cycle)
    return cycle, total_length

def traverse(g, cycle):
    i = 0
    while i < len(cycle) -1:
        print([cycle[i], cycle[i+1]])
        print(g.distance[cycle[i]][cycle[i+1]])
        i+=1
    print([cycle[i], cycle[0]])
    print(g.distance[cycle[i]][cycle[0]])

def break_most_traversed_edge(g, constant):
    index = g.intensity.argmax()
    index = np.unravel_index(index, g.intensity.shape)
    g.distance[index]*=constant
    return index # for logging purposes

distance_matrix = np.array([[0, 10, 15, 20],
                            [10, 0, 35, 25],
                            [15, 35, 0, 30],
                            [20, 25, 30, 0]], dtype='float64')

# Create the Graph instance
graph_instance = Graph(nodes=4, distance=distance_matrix)

# Run the ACO algorithm on this graph instance
best_cycle = ant_colony_optimization(g=graph_instance, verbose=True, iterations=200, ants_per_iteration=500)

# Display the best cycle found and its length
print("Best cycle found:", best_cycle)
print("Best cycle found:", best_so_far)
print("Cycle length:", cycle_length(graph_instance, best_cycle))

import matplotlib.pyplot as plt

plt.plot(range(200), best_so_far, marker='o', linestyle='-', color='b')
plt.xlabel("Iteration")
plt.ylabel("Best So Far")
plt.title("Best So Far vs. Iteration")
plt.grid(True)
plt.show()




#***************************************************************************************************************************************************




import matplotlib.pyplot as plt
import networkx as nx

# Function to plot the best path on the graph
def plot_path(graph, cycle, iteration):
    G = nx.Graph()
    # Add nodes and edges with distances as weights
    for i in range(graph.nodes):
        for j in range(i + 1, graph.nodes):
            G.add_edge(i, j, weight=graph.distance[i][j])
    
    pos = nx.spring_layout(G)  # Layout for visualization
    plt.figure(figsize=(8, 6))
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Highlight the cycle path
    cycle_edges = [(cycle[i], cycle[i+1]) for i in range(len(cycle) - 1)] + [(cycle[-1], cycle[0])]
    nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, width=2.5, edge_color='b', style='solid')
    
    # Add labels for nodes
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    edge_labels = {(i, j): round(graph.distance[i][j], 1) for i in range(graph.nodes) for j in range(i+1, graph.nodes)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title(f"Best Path at Iteration {iteration}")
    plt.show()

# Function to plot pheromone intensity
def plot_pheromone_intensity(graph, iteration):
    plt.figure(figsize=(8, 6))
    plt.imshow(graph.intensity, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Pheromone Intensity')
    plt.title(f"Pheromone Intensity Map at Iteration {iteration}")
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.show()

# Modifications in the ant_colony_optimization function
def ant_colony_optimization(g, verbose=True, iterations=100, ants_per_iteration=50, q=None, degradation_factor=0.9, 
                            use_inertia=False, run_experiment_break=False, run_experiment_artificial_good_cycle=False):
    # Initialization code as before...
    
    for iteration in range(iterations):
        # Existing code for running the algorithm...
        
        # Plot the best path every 50 iterations for visualization
        if iteration % 50 == 0:
            if best_cycle:
                plot_path(g, best_cycle, iteration)
            plot_pheromone_intensity(g, iteration)
    
    # Continue with the rest of the function as is
    return best_cycle




#***************************************************************************************************************************************************




import numpy as np
import matplotlib.pyplot as plt

class Graph:
    def _init_(self, nodes, distance, initial_pheromone_level=None):
        self.nodes = nodes
        self.distance = distance
        assert distance.shape[0] == distance.shape[1]
        
        if initial_pheromone_level:
            self.intensity = np.full_like(distance, initial_pheromone_level, dtype='float64')
        else:
            self.intensity = np.full_like(distance, self.distance.mean() * 10, dtype='float64')

    def _str_(self):
        return f'nodes: {str(self.nodes)}\n{self.distance}\n{self.intensity}'

# Modify the ACO function to include optimal_solution parameter
def ant_colony_optimization(g, verbose=True, iterations=100, ants_per_iteration=50, q=None, 
                            degradation_factor=0.9, optimal_solution=None, initial_pheromone_level=None):
    if q is None:
        q = g.distance.mean()
    
    best_cycle = None
    best_length = float('inf')
    best_so_far = np.zeros(iterations, dtype=np.float64)
    
    for iteration in range(iterations):
        cycles = [traverse_graph(g, random.randint(0, g.nodes - 1)) for _ in range(ants_per_iteration)]
        cycles.sort(key=lambda x: x[1])
        cycles = cycles[:ants_per_iteration // 2]
        
        if best_cycle:
            cycles.append((best_cycle, best_length))
        
        for cycle, total_length in cycles:
            if total_length < best_length:
                best_length = total_length
                best_cycle = cycle

            delta = q / total_length
            for i in range(len(cycle) - 1):
                g.intensity[cycle[i]][cycle[i + 1]] += delta
            g.intensity[cycle[-1]][cycle[0]] += delta
            g.intensity *= degradation_factor
        
        best_so_far[iteration] = best_length

        # Print progress and comparison with optimal solution
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: Best Length = {best_length}")
            if optimal_solution is not None:
                deviation = (best_length - optimal_solution) / optimal_solution * 100
                print(f"Deviation from Optimal: {deviation:.2f}%")

    # Plot the best length vs. iterations
    plt.figure(figsize=(10, 5))
    plt.plot(best_so_far, label="Best Solution Length", color="b")
    if optimal_solution is not None:
        plt.hlines(optimal_solution, 0, iterations, colors="r", linestyles="dashed", label="Optimal Solution")
    plt.xlabel("Iteration")
    plt.ylabel("Cycle Length")
    plt.title("Best Solution Length vs. Optimal Solution")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_cycle, best_length

# Function to perform multiple runs with varying initial pheromone levels
def experiment_with_initial_pheromones(nodes, distance_matrix, optimal_solution):
    initial_pheromone_levels = [distance_matrix.mean() * factor for factor in [5, 10, 20]]
    
    for level in initial_pheromone_levels:
        print(f"\nRunning ACO with Initial Pheromone Level: {level}")
        g = Graph(nodes=nodes, distance=distance_matrix, initial_pheromone_level=level)
        best_cycle, best_length = ant_colony_optimization(g, iterations=200, ants_per_iteration=100, optimal_solution=optimal_solution)
        print(f"Best cycle found: {best_cycle}, Length: {best_length}")

# Define the distance matrix and known optimal solution for a small TSP instance
distance_matrix = np.array([[0, 10, 15, 20],
                            [10, 0, 35, 25],
                            [15, 35, 0, 30],
                            [20, 25, 30, 0]], dtype='float64')
optimal_solution = 80  # Replace with known optimal solution for the specific TSP instance

# Run the experiment with different initial pheromone levels
experiment_with_initial_pheromones(nodes=4, distance_matrix=distance_matrix, optimal_solution=optimal_solution)
