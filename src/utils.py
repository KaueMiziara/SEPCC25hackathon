from typing import cast
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dimod import BinaryQuadraticModel, BINARY
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from scipy.optimize import minimize
from scipy.spatial import distance_matrix


def decodificar_ruta(
    bitstring: str,
    N: int,
    variables_index: dict[str, int],
) -> list[int]:
    """
    Traduce un bitstring '10010...' a una secuencia de nodos visitados.
    Args:
        bitstring (str): Resultado de la medición (ojo: Qiskit usa Little Endian).
        N (int): Número de ubicaciones.
        variables_index (dict): Mapeo {nombre_var: indice_qubit}.
    """
    bits = bitstring[::-1]

    ruta = [-1] * N

    for var_name, idx in variables_index.items():
        if bits[idx] == "1":
            partes = var_name.split("_")
            ubicacion_i = int(partes[1])
            tiempo_t = int(partes[2])

            if tiempo_t < N:
                ruta[tiempo_t] = ubicacion_i

    return ruta


def interpretar_solucion(ansatz, params_optimos, qp):
    """
    Ejecuta el circuito final y busca la solución más probable.
    """
    sampler = StatevectorSampler()

    circuito_medicion = ansatz.copy()
    circuito_medicion.measure_all()

    pub_medicion = (circuito_medicion, [params_optimos])
    job_med = sampler.run([pub_medicion])
    conteos = job_med.result()[0].data.meas.get_counts()

    mejor_bitstring = max(conteos, key=conteos.get)
    probabilidad = conteos[mejor_bitstring] / sum(conteos.values())

    print(f"Estado más probable: {mejor_bitstring} (Prob: {probabilidad:.2%})")

    var_map = {v.name: i for i, v in enumerate(qp.variables)}
    N = int(qp.get_num_vars() ** 0.5)

    ruta_decodificada = decodificar_ruta(mejor_bitstring, N, var_map)

    return ruta_decodificada, conteos


def graficar_convergencia(log_data):
    plt.figure(figsize=(10, 5))
    plt.plot(log_data.conteos, log_data.valores, color="purple", label="Energía VQE")
    plt.xlabel("Iteraciones (Evaluaciones de Costo)")
    plt.ylabel("Energía (Hamiltoniano)")
    plt.title("Convergencia del VQE hacia la Ruta Óptima")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_graph(G, pos):
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color="orange", node_size=800)
    plt.title("Mapa de la Mina")
    plt.show()


def visualizar_ruta_direccionada(G, pos, ruta_optima):
    """
    Dibuja el mapa de la mina resaltando la ruta encontrada con flechas direccionales.

    Args:
        G (nx.Graph): Grafo base de la mina (con todas las conexiones posibles).
        pos (dict): Coordenadas de los nodos.
        ruta_optima (list): Lista ordenada de nodos visitados (ej: [0, 2, 4...]).
        titulo (str): Título del gráfico.
    """
    plt.figure(figsize=(10, 8))

    nx.draw_networkx_nodes(G, pos, node_color="#e0e0e0", node_size=600, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color="#d3d3d3", style="dashed", alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_family="sans-serif", font_weight="bold")

    if -1 in ruta_optima:
        print(
            "¡La ruta contiene nodos inválidos (-1)! No se puede graficar correctamente."
        )
        return

    aristas_ruta = []
    num_pasos = len(ruta_optima)

    print(f"Secuencia de visita: {ruta_optima} -> {ruta_optima[0]} (Ciclo)")

    for i in range(num_pasos):
        origen = ruta_optima[i]
        destino = ruta_optima[(i + 1) % num_pasos]
        aristas_ruta.append((origen, destino))

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=aristas_ruta,
        edge_color="#D9534F",
        width=3.0,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=25,
        connectionstyle="arc3,rad=0.1",
    )

    for i, (u, v) in enumerate(aristas_ruta):
        x_medio = (pos[u][0] + pos[v][0]) / 2
        y_medio = (pos[u][1] + pos[v][1]) / 2

        plt.text(
            x_medio,
            y_medio,
            f"{i + 1}",
            fontsize=10,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.3", fc="#D9534F", ec="none", alpha=0.8),
        )

    nodo_inicio = ruta_optima[0]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[nodo_inicio],
        node_color="#5CB85C",
        node_size=800,
        label="Inicio/Fin",
    )

    plt.title("Ruta Óptima de la Flota", fontsize=14)
    plt.axis("off")

    leyenda_elementos = [
        Line2D([0], [0], color="#D9534F", lw=3, label="Ruta Óptima (VQE)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#5CB85C",
            markersize=12,
            label="Punto de Inicio",
        ),
        Line2D(
            [0], [0], color="#d3d3d3", lw=1, linestyle="--", label="Caminos Disponibles"
        ),
    ]
    plt.legend(handles=leyenda_elementos, loc="lower right")
    plt.show()
