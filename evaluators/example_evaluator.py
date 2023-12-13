from .base import EvaluationBase

class ExampleEvaluator(EvaluationBase):
    def __init__(self):
        super().__init__(
            alpha1=34.73,  # protein_production
            alpha2=49.36,  # protein_production
            alpha3=32.73,  # protein_production
            alpha4=49.54,  # protein_production
            delta1=1.93,  # protein_degradation
            delta2=0.69,  # protein_degradation
            Kd=10.44,  # Kd
            n=4.35  # hill
        )
