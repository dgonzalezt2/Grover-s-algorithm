# Implementation of Grover's Algorithm

This repository contains an implementation of Grover's algorithm in Python using basic matrix operations. No libraries or function packages that implement quantum operations are used. The program receives the problem size (number of qubits) and the oracle function as parameters and generates the circuit that implements Grover's algorithm for that oracle function.

# Description

Grover's algorithm is a quantum search algorithm that provides an efficient way to find a marked element in an unordered list of 2ⁿ elements. The implementation here is done using basic matrix operations to simulate the behavior of quantum circuits.

# Main Functions

1) hadamard_gate()

* Returns the Hadamard gate matrix, a quantum gate that creates an equal superposition of the basis states.

2) tensor_product(A, B)

* Computes the tensor product (Kronecker product) of two matrices, essential for building operators in multi-qubit spaces.

3) identity(n)

* Generates an identity matrix of size 𝑛 × 𝑛 , used as the base for various quantum operators.

4) diffusion_operator(n)

* Creates the diffusion operator used in Grover's algorithm, which amplifies the amplitude of the initial superposition.

5) create_oracle(n, marked_state)

* Creates an oracle that marks a specific state. The oracle inverts the amplitude of the marked state, essential for Grover's search.

6) grover_algorithm(n, oracle)

* Executes Grover's algorithm. It starts with the initial state ∣0⟩ⁿ, applies the initial superposition using Hadamard gates, and then iterates the Grover process (applying the oracle and the diffusion operator) the optimal number of times to amplify the probability of the marked state.

# Code Usage

* Define the number of qubits 𝑛.
* Specify the marked state in binary notation.
* Create the oracle using the `create_oracle` function.
* Execute Grover's algorithm with `grover_algorithm`.
* Print the probabilities of each basis state after running the algorithm.

# Example Usage
```
import numpy as np

def hadamard_gate():
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def tensor_product(A, B):
    return np.kron(A, B)

def identity(n):
    return np.eye(n)

def diffusion_operator(n):
    psi = np.ones((2**n, 2**n)) / (2**n)
    return 2 * psi - identity(2**n)

def create_oracle(n, marked_state):
    oracle = identity(2**n)
    index = int(marked_state, 2)
    oracle[index, index] = -1
    return oracle

def grover_algorithm(n, oracle):
    initial_state = np.zeros((2**n, 1))
    initial_state[0] = 1

    H = hadamard_gate()
    Hn = H
    for _ in range(n-1):
        Hn = tensor_product(Hn, H)

    state = np.dot(Hn, initial_state)
    iterations = int(np.pi / 4 * np.sqrt(2**n))

    for _ in range(iterations):
        state = np.dot(oracle, state)
        D = diffusion_operator(n)
        state = np.dot(D, state)

    return state

# Problem size (number of qubits)
n = 5

# Marked state
marked_state = "0011"

# Oracle
oracle = create_oracle(n, marked_state)

# Execute Grover's algorithm
result = grover_algorithm(n, oracle)

# Print probabilities
probabilities = np.abs(result)**2
for i in range(len(probabilities)):
    print(f"State |{i:0{n}b}>: Probability = {probabilities[i][0]:.4f}")
```
# Requirements

* Python 3.x
* NumPy

# Installation

# Clone the repository
```
git clone https://github.com/dgonzalezt2/Grover-s-algorithm.git
```

# Navigate to the project directory
```
cd Grover-s-algorithm
```

# Install the dependencies
```
pip install numpy
```

# By David Gonzalez Tamayo
<h3 align="left">Connect with me:</h3>
<p align="left">
  <a href="https://www.linkedin.com/in/david-gonz%C3%A1lez-tamayo/" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/de91fca307a83d75fc5b1f6ce24540454acead41/icons/LinkedIn.svg" alt="Linkedin" width="40" height="40"/>
  </a>
    <a href="https://www.instagram.com/davidgonza0326/" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/e67133bc60d96561bc247dfbc3eece0a897285c8/icons/Instagram.svg" alt="Instagram" width="40" height="40"/>
  </a>
</p>
