from typing import cast
import numpy as np
from dimod import BinaryQuadraticModel, BINARY
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo


def crear_hamiltoniano(
    dist_matrix: np.ndarray, A: float | None = None, B: float = 1.0
) -> BinaryQuadraticModel:
    """
    Construye el modelo BQM  para el problema de rutas.

    Args:
        dist_matrix (np.ndarray): Matriz N x N de distancias (con penalizaciones).
        A (float): Penalización por romper restricciones (Unicidad).
        B (float): Peso de la función de costo (Distancia).

    Returns:
        bqm (BinaryQuadraticModel): El modelo listo para resolver.
    """
    N = len(dist_matrix)

    max_dist = np.max(dist_matrix)
    if A is None:
        A = B * max_dist * 2.0
        print(f"Constante A calculada automáticamente: {A:.2f} (Max dist: {max_dist})")
    A = cast(float, A)

    bqm = BinaryQuadraticModel(BINARY)

    get_var = lambda i, t: f"x_{i}_{t}"

    for i in range(N):
        for t in range(N):
            var_name = get_var(i, t)
            bqm.add_variable(var_name, -2 * A)

    for t in range(N):
        for i in range(N):
            for j in range(i + 1, N):
                u = get_var(i, t)
                v = get_var(j, t)
                bqm.add_interaction(u, v, 2 * A)

    for i in range(N):
        for t in range(N):
            for k in range(t + 1, N):
                u = get_var(i, t)
                v = get_var(i, k)
                bqm.add_interaction(u, v, 2 * A)

    for t in range(N):
        next_t = (t + 1) % N

        for i in range(N):
            for j in range(N):
                if i != j:
                    dist = dist_matrix[i][j]

                    u = get_var(i, t)
                    v = get_var(j, next_t)

                    bqm.add_interaction(u, v, B * dist)

    return bqm


def bqm_to_qp(bqm: BinaryQuadraticModel) -> QuadraticProgram:
    """
    Traduce un BinaryQuadraticModel de dimod a un QuadraticProgram de Qiskit.
    """
    qp = QuadraticProgram(name="Optimización de Minería")

    for var in bqm.variables:
        qp.binary_var(name=str(var))

    linear_terms = {str(k): v for k, v in bqm.linear.items()}

    quadratic_terms = {(str(u), str(v)): val for (u, v), val in bqm.quadratic.items()}

    qp.minimize(linear=linear_terms, quadratic=quadratic_terms)

    return qp


def qp_to_ising(qp: QuadraticProgram) -> tuple[SparsePauliOp, float]:
    """
    Converte un QuadraticProgram a un problema de Ising.
    """
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)
    ising = qubo.to_ising()

    return ising


def preparar_problema(
    bqm: BinaryQuadraticModel,
) -> tuple[SparsePauliOp, QuantumCircuit, float]:
    """
    Convierte el BinaryQuadraticModel a un operador Ising y define el Ansatz.
    """
    qp = bqm_to_qp(bqm)
    hamiltoniano, offset = qp_to_ising(qp)

    ansatz = real_amplitudes(
        num_qubits=hamiltoniano.num_qubits,
        entanglement="linear",
        reps=1,
    )
    ansatz = ansatz.decompose()

    print(f"Problema preparado: {hamiltoniano.num_qubits} Qubits.")
    print(f"Parámetros del Ansatz: {ansatz.num_parameters}")

    return hamiltoniano, ansatz, offset
