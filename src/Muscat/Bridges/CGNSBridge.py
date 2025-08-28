class MeshToCGNS:  # pragma: no cover - minimal stub for tests
    def __init__(self, mesh=None):
        self.mesh = mesh

    def __call__(self, mesh):
        return self.__class__(mesh)

