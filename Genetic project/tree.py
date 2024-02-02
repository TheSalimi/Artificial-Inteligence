from enum import Enum
import random as rnd
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
from tree import *
import copy
from sklearn.metrics import mean_squared_error

operators = {'+': np.add, '-': np.subtract, '*': np.multiply, '/': np.divide, '^': np.power, 'sin': np.sin, 'cos': np.cos}

class Type(Enum):
    Operator = 0
    Operand = 1

def operands_count(operator):
    if operator =='sin' or operator == 'cos':
        return 1
    else:
        return 2

class Learn:
    def __init__(self, dimensions=2, max_depth=10, population_size=300, generations=30, mutation_rate=0.1):
        self.generations = generations
        self.population = []
        self.dimensions = dimensions
        self.max_depth = max_depth
        self.mutation_rate = mutation_rate
        self.population_size = population_size

    def init_pop(self):
        for _ in range(self.population_size):
            tree = Tree.make_random_tree(rnd.randint(0, self.max_depth), self.dimensions)
            self.population.append(tree)

    def measure_fitness(self, x, y, z=None):
        score = []
         
        for tree in self.population:
            predicted = tree.measure({'x': x, 'y': y})

            if type(predicted) == int:
                predicted = np.full(x.shape, predicted)
            try:
                fitness = mean_squared_error(y if z is None else z, predicted)
            except:
                fitness = float('inf')

            score.append(fitness)

        return score

    def selection(self, fitness_scores):
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k])
        selected_parents = [self.population[i] for i in sorted_indices[:self.population_size]]

        return selected_parents
        
    def crossover(self, parent1, parent2):
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        crossover_node_parent1 = random.choice(parent1.return_nodes())
        crossover_node_parent2 = random.choice(parent2.return_nodes())

        crossover_node_parent1.replace_subtree(child2)
        crossover_node_parent2.replace_subtree(child1)

        return child1, child2

    def mutation(self, tree):
        mutation_node = random.choice(tree.return_nodes())
        new_subtree = Tree.make_random_tree(random.randint(1, self.max_depth))
        mutation_node.replace_subtree(new_subtree.root)

    def fit(self, x, y, z=None):
        self.init_pop()
        rows = self.generations // 5
        
        if z is None:
            figure, axis = plt.subplots(rows, 5)
        else:
            figure, axis = plt.subplots(rows, 5, subplot_kw={'projection': '3d'})
        
        average_mse_before = float('inf')
        
        for i in range(rows):   
            fitness_scores = self.measure_fitness(x, y, z)                   
            stillness_counter = 1 
            for j in range(5):
                parents = self.selection(fitness_scores)
                average_mse_current = np.average(fitness_scores)

                if (average_mse_current < average_mse_before and stillness_counter <= 5):
                    stillness_counter += 1
                else:
                    stillness_counter = 1

                offspring = []

                for q in range(1, len(parents)):
                    parent1 = parents[q]
                    parent2 = parents[q - 1]
                    child1, child2 = self.crossover(parent1, parent2)
                    offspring.extend([child1, child2])

                for tree in offspring:
                    random = rnd.random()

                    if random < self.mutation_rate * stillness_counter:
                        self.mutation(tree)

                self.population = offspring

                best_index = np.argmax(fitness_scores)
                best_fitness = fitness_scores[best_index]
                best_tree = self.population[best_index]
                
                predicted = best_tree.measure({'x': x, 'y': y})

                if type(predicted) == int:
                    predicted = np.full(x.shape, predicted)
                    
                if z is None:
                    axis[i, j].plot(x, y, 'blue')
                    axis[i, j].plot(x, predicted, 'orange')
                else:
                    axis[i, j].plot3D(x, y, z, 'blue')
                    axis[i, j].plot3D(x, y, predicted, 'orange')
                
                generation = i * rows + j + 1
                axis[i, j].set_title(f"{generation}")
                print(f"generation = {generation} =>, best fitness : {best_fitness}")
                
                if best_fitness == 0.0:
                    return best_tree
            
        figure.show()

        best_index = np.argmax(fitness_scores)

        return self.population[best_index] 

class Node:
    def __init__(self, type: Type, data):
        self.parent = None
        self.left = None
        self.right = None
        self.type = type
        self.data = data
        
    def make_random_node(type: Type, terminal: bool=False, not_zero : bool=False, not_negative: bool=False, dimensions: int=2):
        if type == Type.Operand:

            if not terminal:
                data = rnd.randint(-5, 5)
                
                if (not_negative and data <= 0):
                    data = rnd.randint(1, 5)
                elif (not_zero and data == 0 ):
                    random_choice = rnd.randint(1, 2)
                    data = rnd.randint(-5, -1) if random_choice == 1 else rnd.randint(1, 5)
            else:
                return Node(type, ['x','y'][rnd.randint(0, dimensions - 2)])
        else:
            data = rnd.choice(list(operators.keys()))
        
        return Node(type, data)
        
    def replace_subtree(self, new_subtree_root):
        if self.parent is not None:
            if self.parent.left == self:
                self.parent.left = new_subtree_root
            elif self.parent.right == self:
                self.parent.right = new_subtree_root

        if new_subtree_root is not None:
            new_subtree_root.parent = self.parent

        self.parent = None

class Tree:
    def __init__(self, root: Node=None):
        self.root = root
    
    def make_random_tree(depth, dimensions=2):
        if depth != 0:
            root = Node.make_random_node(Type.Operator)
            Tree.create_rand_tree(depth - 1, root, dimensions)
            return Tree(root)
        else:
            return Tree(Node.make_random_node(Type.Operand))
    
    def create_rand_tree(depth, parent: Node=None, dimensions=2):
        op_count = operands_count(parent.data)

        if (depth == 0):
            parent.right = Node.make_random_node(Type.Operand, op_count == 1, parent.data == '/', parent.data == '^', dimensions)

            if (operands_count(parent.data) != 1):
                parent.left = Node.make_random_node(Type.Operand, True)
        else:                    
            type = Type.Operator if rnd.random() < 0.5 else Type.Operand
            parent.right = Node.make_random_node(type, op_count == 1, parent.data == '/', parent.data == '^', dimensions)
            
            if (op_count != 1):
                type = Type.Operator if rnd.random() < 0.5 else Type.Operand
                parent.left = Node.make_random_node(type, True, dimensions=dimensions)
            
            if (parent.right.type == Type.Operator):
                Tree.create_rand_tree(depth - 1, parent.right, dimensions)
                
            if (operands_count(parent.data) != 1 and parent.left.type == Type.Operator):
                Tree.create_rand_tree(depth - 1, parent.left, dimensions)
            
    def measure(self, dic: dict, node: Node=None):
        if node is None:
            node = self.root

        if (type(node.data) == int):
            return node.data
        
        if (node.data in dic.keys()):
            return dic[node.data]
        
        if (node.right):
            right_result = self.measure(dic, node.right)

        if (node.left):
            left_result = self.measure(dic, node.left)

        if (operands_count(node.data) == 1):
            return operators[node.data](right_result)
        else:
            return operators[node.data](left_result, right_result)    

    def return_nodes(self):
        return self.__return_nodes(self.root)

    def __return_nodes(self, node):
        if node is None:
            return []
        return [node] + self.__return_nodes(node.left) + self.__return_nodes(node.right)

    def traverse(self, node: Node=None):
        if node is None:
            return ''
        
        if node is not None:
            left = self.__traverse(node.left)
            right = self.__traverse(node.right)
            return '(' + left + ' ' + str(node.data) + ' ' + right + ')'