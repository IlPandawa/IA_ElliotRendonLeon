class ModelInterface:
    
    def __init__(self):
        self.is_trained = False
        self.model_name = "Modelo Base"
    
    def train(self, data):
        raise NotImplementedError("Las subclases deben implementar train()")
    
    def predict(self, velocidad_bala, distancia):
        raise NotImplementedError("Las subclases deben implementar predict()")
    
    def load(self):
        raise NotImplementedError("Las subclases deben implementar load()")
    
    def save(self):
        raise NotImplementedError("Las subclases deben implementar save()")