import numpy as np

def hadamard_gate():
    """
    Crea la matriz de la puerta de Hadamard.

    Returns:
        numpy.ndarray: Matriz de la puerta de Hadamard.
    """
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def tensor_product(A, B):
    """
    Calcula el producto tensorial (producto de Kronecker) de dos matrices.

    Args:
        A (numpy.ndarray): Primera matriz.
        B (numpy.ndarray): Segunda matriz.

    Returns:
        numpy.ndarray: Producto tensorial de A y B.
    """
    return np.kron(A, B)

def identity(n):
    """
    Crea una matriz identidad de tamaño n x n.

    Args:
        n (int): Dimensión de la matriz identidad.

    Returns:
        numpy.ndarray: Matriz identidad de tamaño n x n.
    """
    return np.eye(n)

def diffusion_operator(n):
    """
    Crea el operador de difusión para el algoritmo de Grover.

    Args:
        n (int): Número de qubits.

    Returns:
        numpy.ndarray: Operador de difusión de tamaño (2^n) x (2^n).
    """
    # Crear el operador de proyección |\psi><\psi|
    psi = np.ones((2**n, 2**n)) / (2**n)
    
    # Construir el operador de difusión 2|\psi><\psi| - I
    return 2 * psi - identity(2**n)

def create_oracle(n, marked_state):
    """
    Crea un oráculo para el algoritmo de Grover que marca un estado específico.

    Args:
        n (int): Número de qubits.
        marked_state (str): Estado marcado en representación binaria.

    Returns:
        numpy.ndarray: Oráculo de tamaño (2^n) x (2^n) que marca el estado especificado.
    """
    # Inicializar el oráculo como la matriz identidad
    oracle = identity(2**n)
    
    # Convertir el estado marcado a un índice
    index = int(marked_state, 2)
    
    # Invertir la amplitud del estado marcado
    oracle[index, index] = -1
    
    return oracle

def grover_algorithm(n, oracle):
    """
    Implementa el algoritmo de Grover para la búsqueda de un estado marcado.

    Args:
        n (int): Número de qubits.
        oracle (numpy.ndarray): Oráculo que marca el estado objetivo.

    Returns:
        numpy.ndarray: Estado final después de aplicar el algoritmo de Grover.
    """
    # Estado inicial |0>^n
    initial_state = np.zeros((2**n, 1))
    initial_state[0] = 1
    
    # Aplicar Hadamard a cada qubit para crear superposición
    H = hadamard_gate()
    Hn = H
    for _ in range(n-1):
        Hn = tensor_product(Hn, H)
    
    state = np.dot(Hn, initial_state)
    
    # Número óptimo de iteraciones de Grover
    iterations = int(np.pi / 4 * np.sqrt(2**n))
    
    # Iteraciones de Grover
    for _ in range(iterations):
        # Aplicar el oráculo
        state = np.dot(oracle, state)
        # Aplicar el operador de difusión
        D = diffusion_operator(n)
        state = np.dot(D, state)
    
    return state

# Tamaño del problema (número de qubits)
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
