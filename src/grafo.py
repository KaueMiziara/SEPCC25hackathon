import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix

def crear_grafo_mina(n_nodes, n_bloqueos=1):
    """
    Crea un grafo geométrico donde algunos caminos están bloqueados.
    Los pesos de las aristas representan la distancia euclidiana.

    Retorna la matriz de distancias lista para el Hamiltoniano.
    """

    coords = np.random.rand(n_nodes, 2) * 100
    pos = {i: coords[i] for i in range(n_nodes)}

    dist_mat = distance_matrix(coords, coords)

    G = nx.complete_graph(n_nodes)
    nx.set_node_attributes(G, pos, "pos")

    lista_aristas = list(G.edges())
    indices_a_bloquear = np.random.choice(
        len(lista_aristas), size=n_bloqueos, replace=False
    )

    for i in indices_a_bloquear:
        u, v = lista_aristas[i]

        G.remove_edge(u, v)
        PENALTY = 9999.0
        dist_mat[u][v] = PENALTY
        dist_mat[v][u] = PENALTY

        print(f"> Bloqueo generado entre Ubicación {u} y {v}")

    for u, v in G.edges():
        G[u][v]["weight"] = dist_mat[u][v]

    return G, dist_mat, pos