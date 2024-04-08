from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

bayesNet = BayesianNetwork()
bayesNet.add_node("A")
bayesNet.add_node("B")
bayesNet.add_node("C")
bayesNet.add_node("D")

bayesNet.add_edge("A", "B")
bayesNet.add_edge("A", "C")
bayesNet.add_edge("B", "C")
bayesNet.add_edge("C", "D")

cpd_A = TabularCPD('A', 3, values=[[.10], [.20], [.70]])
cpd_B = TabularCPD('B', 2, values=[[.7, 0.5, 0.2], [.3, 0.5, 0.8]], evidence= ['A'], evidence_card=[3])
cpd_C = TabularCPD('C', 2, values=[[.90, 0.5, 0.9, 0.3, 0.7, 0.1], [.10, 0.5, 0.1, 0.7, 0.3, 0.9]], evidence= ['A', 'B'], evidence_card=[3, 2])
cpd_D = TabularCPD('D', 2, values=[[0.8, .2], [.2, .8]], evidence=['C'], evidence_card=[2])

bayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_D)
bayesNet.check_model()
print("Model is correct.")
solver = VariableElimination(bayesNet)
result = solver.query(variables=['D'])
print(result)
result = solver.query(variables=['D'], evidence={'C': 1})
print("\nD| C:\n", result)

