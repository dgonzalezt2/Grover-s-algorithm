import numpy as np

def hadamard_gate():

    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def tensor_product(A, B):

    return np.kron(A, B)

def identity(n):

    return np.eye(n)

def diffusion_operator(n):

    # Crea la proyección |\psi><\psi|
    psi = np.ones((2**n, 2**n)) / (2**n)
    
    # Construye la difusión 2|\psi><\psi| - I
    return 2 * psi - identity(2**n)

def create_oracle(n, marked_state):

    # Inicia el oraculo en la matriz identidad
    oracle = identity(2**n)
    
    # Convirte el indice marcado
    index = int(marked_state, 2)
    
    # Invierta la amplitud
    oracle[index, index] = -1
    
    return oracle

def grover_algorithm(n, oracle):

    # Estado inicial |0>^n
    initial_state = np.zeros((2**n, 1))
    initial_state[0] = 1
    
    # Aplica Hadamard a cada qubit para crear la superposición
    H = hadamard_gate()
    Hn = H
    for _ in range(n-1):
        Hn = tensor_product(Hn, H)
    
    state = np.dot(Hn, initial_state)
    
    # Número óptimo de iteraciones de Grover
    iterations = int(np.pi / 4 * np.sqrt(2**n))
    
    # Iteraciones de Grover
    for _ in range(iterations):
        # Aplica el oráculo
        state = np.dot(oracle, state)
        # Aplica el operador de difusión
        D = diffusion_operator(n)
        state = np.dot(D, state)
    
    return state

# Tamaño del problema (# de qubits)
n = 5

# Estado marcado (cambiar esto para marcar un estado diferente)
marked_state = "0011"

# Oráculo
oracle = create_oracle(n, marked_state)

# Ejecutar el algoritmo de Grover
result = grover_algorithm(n, oracle)

# Imprimir probabilidades
probabilities = np.abs(result)**2
for i in range(len(probabilities)):
    print(f"Estado |{i:0{n}b}>: Probabilidad = {probabilities[i][0]:.4f}")
