# Genetic tree

## Introduction
This Python project implements a symbolic regression algorithm using genetic programming. It utilizes a tree-based representation of mathematical expressions, evolving them to fit given data through generations. The algorithm employs operators such as addition, subtraction, multiplication, division, exponentiation, sine, and cosine. It incorporates a population-based approach with crossover and mutation operations to evolve solutions towards minimizing mean squared error. The Learn class orchestrates the evolutionary process, including selection, crossover, and mutation. The Tree class represents and manipulates mathematical expression trees, facilitating the generation and evaluation of candidate solutions. The project aims to autonomously discover mathematical expressions that accurately model relationships between input variables and target values, showcasing the power of genetic programming in symbolic regression tasks

## main.ipynb
you can see diffrenct phases and results in main.ipynb

#### part 1 : 
approximate the function that maps the input values to the output values

###### cos(x) :
![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/f2e664f9-8913-430a-9c04-451929f7a5c0)
![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/b032e5ba-2d9b-42da-b3ee-690c5f425c95)

###### sin(x) + cos(x) :
![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/0674d37b-705e-4833-ba8e-d94bdcdadb06)
![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/24a39134-6db3-4f1a-87d5-088ace4be468)

###### sin^2(x)
![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/bfeebac9-1139-43a9-ad67-16066832e5a4)
![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/5836b524-a371-49f1-a3bd-32064d96a635)

###### arctan(x)
![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/ee825104-be74-494f-a7cb-a5fb8baa6cc9)
![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/cae1b67e-38e0-4e05-a926-8cb0c05c6488)

#### part 2 :
use 3 different function

```python
def random_function(x):
    result = np.zeros_like(x)
    
    result[x < -5] = (x[x < -5])
    result[(x >= -5) & (x <= 5)] = np.arctan(x[(x >= -5) & (x <= 5)])
    result[x > 5] = x[x > 5]*(3/2)
    
    return result

x = np.linspace(-10, 10, 1000)
y = random_function(x)
g = Learn()
print(g.fit(x, y).traverse())
```
![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/5f00570d-1cca-49ef-83ed-a1ab5cda9094)
![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/5f56726b-0a3a-4cce-ac6c-8d5f32c84eaa)

#### part 3 :
3 dimensions

```python
x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)
z = x * 2 + y
g = Learn(dimensions=3)
print(g.fit(x, y, z).traverse())
```

![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/29af75ee-9274-4e9b-a7d9-f77bdcaebae3)
![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/ef19a728-f6e7-4714-8072-fe23cab5b33b)


