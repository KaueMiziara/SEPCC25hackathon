class LogVQE:
    def __init__(self) -> None:
        self.conteos = []
        self.valores = []
        self.parametros = []
        self.iteracion = 0

    def update(self, xk):
        """Callback que llama SciPy en cada iteración."""
        self.iteracion += 1
        self.parametros.append(xk)

    def log_energia(self, energia):
        """Método auxiliar para guardar la energía desde la función de costo."""
        self.valores.append(energia)
        self.conteos.append(len(self.valores))
