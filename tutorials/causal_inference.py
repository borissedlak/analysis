from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import CausalInference

model = BayesianNetwork([("X", "Y"), ("Z", "X"), ("Z", "W"), ("W", "Y")])
cpd_z = TabularCPD(variable="Z", variable_card=2, values=[[0.2], [0.8]])

cpd_x = TabularCPD(
    variable="X",
    variable_card=2,
    values=[[0.1, 0.3], [0.9, 0.7]],
    evidence=["Z"],
    evidence_card=[2],
)

cpd_w = TabularCPD(
    variable="W",
    variable_card=2,
    values=[[0.2, 0.9], [0.8, 0.1]],
    evidence=["Z"],
    evidence_card=[2],
)

cpd_y = TabularCPD(
    variable="Y",
    variable_card=2,
    values=[[0.3, 0.4, 0.7, 0.8], [0.7, 0.6, 0.3, 0.2]],
    evidence=["X", "W"],
    evidence_card=[2, 2],
)

model.add_cpds(cpd_z, cpd_x, cpd_w, cpd_y)

# Do operation with a specified adjustment set.
# infer = CausalInference(model)
# do_X_W = infer.query(["Y"], do={"X": 1}, adjustment_set=["W"])
# print(do_X_W)
#
# do_X_Z = infer.query(["Y"], do={"X": 1}, adjustment_set=["Z"])
# print(do_X_Z)
#
# do_X_WZ = infer.query(["Y"], do={"X": 1}, adjustment_set=["W", "Z"])
# print(do_X_WZ)

# Adjustment without do operation.
infer = CausalInference(model)
# adj_W = infer.query(["Y"], adjustment_set=["W"])
# print(adj_W)
#
# adj_Z = infer.query(["Y"], adjustment_set=["Z"])
# print(adj_Z)

adj_WZ = infer.query(["Y", "X"])
print(adj_WZ)

infer_non_adjust = VariableElimination(model)
print(infer_non_adjust.query(variables=["X"], evidence={"Y": 1}))
print(infer_non_adjust.query(variables=["X"]))
