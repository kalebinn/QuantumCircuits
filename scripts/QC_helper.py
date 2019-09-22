# =============================================================================
# this python file is a collection of helper functions to draw the quantum
# for various datasets used in quantum support vector machines by qiskit
# this file was created by Kelvin Ma

# the quantum support vector machine has three general steps:
#   1. Create the feature map. There are multiply ways to create a feature map
#       In most cases, we will be using the Second Order Expansion method. This
#       is done on a classical machine
#   2. Caluclate the Kernel Matrix (on a Quantum Computer).
#      The kernal matrix is a tensor product of
#       every data pair (feature data pairs). For example the first element
#       of the kernel matrix is: x(0).inverse * x(0) in the new feature map.
#   3. The QSVM algorithm contains the feature map as a member after the
#       constructor is called.
#   4. This method is known as kernalization (it optimizes the cost)
# =============================================================================
import math
#from datasets import * # dont need this here
import numpy as np

# import the neccessary modules for quantum support vector machine
from qiskit import BasicAer
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.feature_maps import SecondOrderExpansion

# At the time of writing, Qiskit only supports 2-3 features.
def prepare_data_set_for_qc(num_circuits, class_A_dataset, class_B_dataset,
                     class_C_dataset = None, start_index = 0):
    if class_C_dataset is None:
        num_classes = 2
        num_data_per_class = int(math.sqrt(num_circuits)/num_classes)
        end_index = start_index + num_data_per_class
        slice_A = class_A_dataset[start_index:end_index]
        slice_B = class_B_dataset[start_index:end_index]
        data_for_qc = np.concatenate((slice_A,slice_B), axis = 0)
        return data_for_qc

    num_classes = 3
    num_data_per_class = int(math.sqrt(num_circuits)/num_classes)
    end_index = start_index + num_data_per_class
    slice_A = class_A_dataset[start_index:end_index]
    slice_B = class_B_dataset[start_index:end_index]
    slice_C = class_C_dataset[start_index:end_index]
    data_for_qc = np.concatenate((slice_A, slice_B, slice_C), axis = 0)
    return data_for_qc

def draw_circuits(qsvm, dataset, print_circuit_info = True, add_measurement = False,
                  jupyter_notebook = False):
    circuit_count = 0
    for i in range(0,len(dataset)):
        for j in range (0,len(dataset)):
            quantum_circuit = qsvm.construct_circuit(dataset[i], dataset[j],
                                                     measurement = add_measurement)
            print(f'Circuit for: x({i}) transpose * x({j})')
            if (print_circuit_info):
                print('circuit info:\n\tcircuit depth = ',
                      quantum_circuit.depth(), '\tcircuit width = ',
                      quantum_circuit.width())
                print('Circuit operation breakdown:\n\t',
                      quantum_circuit.count_ops())
            if (jupyter_notebook):
                print(quantum_circuit.draw(output = 'mpl'))
            else:
                print(quantum_circuit.draw())
            print('\n\n')
            circuit_count += 1
    return None

    print(circuit_count , " circuits were drawn.")
    return circuits
