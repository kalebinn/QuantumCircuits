# Python script for retrieving quantum circuits for ad_hoc_data

from datasets import *

from qiskit import BasicAer
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.input import ClassificationInput
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.feature_maps import *

feature_dim = 2

sample_Total, training_input, test_input, class_labels = ad_hoc_data(
    training_size = 20,
    test_size = 10,
    n = feature_dim,
    gap = 0.3,
    PLOT_DATA = True
)

extra_test_data = sample_ad_hoc_data(sample_Total, 10, n = feature_dim)
datapoints, class_to_label = split_dataset_to_data_and_labels(extra_test_data)

print(class_to_label)

seed = 50594

feature_map = SecondOrderExpansion(feature_dimension = feature_dim,
                                  depth = 2,
                                  entanglement = 'full')

qsvm = QSVM(feature_map, training_input, test_input, datapoints[0])
backend = BasicAer.get_backend ('qasm_simulator')
quantum_instance = QuantumInstance (backend, shots = 1024, seed = seed, seed_transpiler = seed)

qsvm_results = qsvm.run(quantum_instance)

#print(training_input.get('A'))
A_datapoints = training_input.get('A')
B_datapoints = training_input.get('B')

# constructing the quantum circuits from the data (and feature map)
print("Class label A")
print("Class label data:")
print(A_datapoints)
print("Class label A quantum circuit")
qc_A = qsvm.construct_circuit(A_datapoints[0], A_datapoints[1], measurement = True)
print(qc_A.draw())
print("Circuit Depth: ", qc_A.depth(), "\tCircuit Width: ", qc_A.width())

print("\n\n\nClass label B")
print("Class label data:")
print(B_datapoints)
print("Class label B quantum circuit")
qc_B = qsvm.construct_circuit(B_datapoints[0], B_datapoints[1], measurement = True)
print(qc_B.draw())
print("Circuit Depth: ", qc_B.depth(), "\tCircuit Width: ", qc_B.width())

