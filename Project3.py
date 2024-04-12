from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

# Creates all of the nodes in the Bayesian Network 
bayesNet = BayesianNetwork()
bayesNet.add_node("A")
bayesNet.add_node("B")
bayesNet.add_node("C")
bayesNet.add_node("D")

# Adds the appropriate edges to all of the nodes within the network.
bayesNet.add_edge("A", "B")
bayesNet.add_edge("A", "C")
bayesNet.add_edge("B", "C")
bayesNet.add_edge("C", "D")

# Create each of the probability tables for each node.  False/Negative values are given before True/Positive values.
cpd_A = TabularCPD('A', 3, values=[[.10], [.20], [.70]])
cpd_B = TabularCPD('B', 2, values=[[.7, 0.5, 0.2], [.3, 0.5, 0.8]], evidence= ['A'], evidence_card=[3]) # B is depentent on A
cpd_C = TabularCPD('C', 2, values=[[.90, 0.5, 0.9, 0.3, 0.7, 0.1], [.10, 0.5, 0.1, 0.7, 0.3, 0.9]], evidence= ['A', 'B'], evidence_card=[3, 2]) # C is dependent on A and B
cpd_D = TabularCPD('D', 2, values=[[0.8, .2], [.2, .8]], evidence=['C'], evidence_card=[2]) # D is dependent on C
cpd_D2 = TabularCPD('D', 2, values=[[0.8, .8, .8, .8, 0.8, .8, 0.2, .2, 0.2, .2,0.2, .2],[0.2, .2, .2, .2, 0.2, .2, 0.8, .8, 0.8, .8,0.8, .8]], evidence=['C','A','B'], evidence_card=[2,3,2])
# cpd_D2 shows how changing A and B will affect C, which in turn will directly affect D.

# Creates the model of the network
bayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_D)
bayesNet.check_model()
print("Model is correct.")
solver = VariableElimination(bayesNet)
print(bayesNet)

# Solves each of the Queries  given in the Project.
query1 = solver.query(variables=['A'])
query2 = solver.query(variables=['B'])
query3 = solver.query(variables=['C'], evidence={'A': 1, 'B': 0}) # A is Fair, B is Negative
query4 = solver.query(variables=['D'], evidence={'C': 0}) # C is Not Treated
query5 = solver.query(variables=['D'])
query7 = cpd_D
query6 = cpd_C
query8= cpd_D2

# Prints the Queries out to the terminal, so that they may be viewed.
print("Query 1: A", query1)
print("\nQuery 2: B:\n", query2)
print("\nQuery 3: C| A, B:\n", query3)
print("\nQuery 4: D| C,A,B:\n", query4)
print("\nQuery 5: D\n", query5)
print("\nQuery 6: D\n", query6)
print("\nQuery 7: D\n", query7)
print("\nQuery 8: D\n", query8)


