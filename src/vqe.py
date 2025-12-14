import numpy as np
from dimod import BinaryQuadraticModel
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

from .log_vqe import LogVQE
from .modelo import preparar_problema


def run_vqe(bqm: BinaryQuadraticModel, log: LogVQE):
    hamiltoniano, ansatz, offset = preparar_problema(bqm)

    estimator = StatevectorEstimator()

    num_params = ansatz.num_parameters
    init_params = 2 * np.pi * np.random.rand(num_params)

    def cost_func_wrapper(params):
        params_flat = np.array(params).flatten()

        if len(params_flat) != ansatz.num_parameters:
            raise ValueError(
                f"Desajuste: Ansatz pide {ansatz.num_parameters}, recibí {len(params_flat)}"
            )

        pub = (ansatz, [hamiltoniano], [params_flat])

        job = estimator.run([pub])
        result = job.result()
        energia = result[0].data.evs[0]

        log.log_energia(energia)
        return energia

    print("Iniciando optimización VQE con COBYLA")

    res = minimize(
        cost_func_wrapper,
        init_params,
        method="COBYLA",
        options={"maxiter": 500, "disp": True},
    )

    energia_final = res.fun + offset

    return res, log, ansatz, energia_final
