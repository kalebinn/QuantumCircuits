# Python script for retrieving quantum circuits for ad_hoc_data

from datasets import *
import math 
from QC_helper import *
from qiskit import BasicAer
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.input import ClassificationInput
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.feature_maps import *

# helper functions

# QSVM
feature_dim = 2

# =============================================================================
# sample_Total, training_input, test_input, class_labels = ad_hoc_data(
#     training_size = 20,
#     test_size = 10,
#     n = feature_dim,
#     gap = 0.3,
#     PLOT_DATA = True
# )
# 
# extra_test_data = sample_ad_hoc_data(sample_Total, 10, n = feature_dim)
# datapoints, class_to_label = split_dataset_to_data_and_labels(extra_test_data)
# 
# print(class_to_label)
# 
# seed = 50594
# 
# feature_map = SecondOrderExpansion(feature_dimension = feature_dim,
#                                   depth = 2,
#                                   entanglement = 'full')
# 
# qsvm = QSVM(feature_map, training_input, test_input, datapoints[0])
# backend = BasicAer.get_backend ('qasm_simulator')
# quantum_instance = QuantumInstance (backend, shots = 1024, seed = seed, seed_transpiler = seed)
# 
# qsvm_results = qsvm.run(quantum_instance)
# =============================================================================

# circuit extraction
# let's say we only have want 4 circuits
num_circuits = 4
# let's extract the datapoints from the training data first:
training_data_A = training_input.get('A')
training_data_B = training_input.get('B')

data_for_qc = prepare_data_set_for_qc(num_circuits,
                                      training_data_A, 
                                      training_data_B)
draw_circuits(qsvm,data_for_qc)

