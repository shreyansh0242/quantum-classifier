# Diabetes Classification - QNN model
# ---------------------------------------------------------------------------- #
## Update package resources to account for version changes.
import importlib, pkg_resources
importlib.reload(pkg_resources)

# ---------------------------------------------------------------------------- #
## Importing essential Header Files
import cv2, os, cirq, sympy, collections, os
import numpy as np
import seaborn as sns
import pandas as pd

import tensorflow as tf
import tensorflow_quantum as tfq

import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

import random

# ---------------------------------------------------------------------------- #
## This function loads the dataset into the program
def load_dataset(training_data_path, training_data_labels_path, testing_data_path, testing_data_labels_path, index):
    print("TASK : Loading the Training Dataset")
    x_train = []
    # for image_name in os.listdir(training_data_path):
    #     image_path = f"{training_data_path}/{image_name}"
    #     raw_image = tf.keras.utils.load_img(image_path, target_size=(4,4), color_mode = 'grayscale')
    #     # raw_image = tf.keras.utils.load_img(image_path, target_size=(89,134), color_mode = 'grayscale')
    #     normalized_image = np.array(raw_image)/255.0
    #     x_train.append(normalized_image)
    
    # print(np.array(x_train).shape)

    training_data = pd.read_csv("src/dataset/training_data/training_data.csv")
    # print(training_data)
    for i in range(413):
        # data = np.array(training_data[i])
        data = np.array(training_data[i:i+1])
        data_resize = data.reshape(4,4)
        x_train.append(data_resize)

    x_train = np.array(x_train)

        
    y_train = []
    training_dataset_labels = pd.read_csv(training_data_labels_path)
    for i in range(len(training_dataset_labels)):
        grade = training_dataset_labels['retinopathy_grade'][i]
        # val = 1 if(grade in (1,2,3,0)) else 0
        val = 1 if(grade == index) else 0
        y_train.append(val)

    print("TASK : Loading the Test Dataset")
    x_test = []
    # for image_name in os.listdir(testing_data_path):
    #     image_path = f"{testing_data_path}/{image_name}"
    #     raw_image = tf.keras.utils.load_img(image_path, target_size=(4,4), color_mode = 'grayscale')
    #     # raw_image = tf.keras.utils.load_img(image_path, target_size=(89,134), color_mode = 'grayscale')
    #     normalized_image = np.array(raw_image)/255.0
    #     x_test.append(normalized_image)


    testing_data = pd.read_csv("src/dataset/testing_data/testing_data.csv")
    
    for i in range(103):
        data = np.array(testing_data[i:i+1])
        data_resize = data.reshape(4,4)
        x_test.append(data_resize)

    x_test = np.array(x_test)


    y_test = []
    testing_data_labels = pd.read_csv(testing_data_labels_path)
    for i in range(len(testing_data_labels)):
        grade = testing_data_labels['retinopathy_grade'][i]
        # val = 1 if(grade in (1,2,3,0)) else 0
        val = 1 if(grade == index) else 0
        y_test.append(val)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

# ---------------------------------------------------------------------------- #
## Encode the data as quantum circuits
def convert_to_circuit(image):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    # qubits = cirq.GridQubit.rect(8, 8)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


def quantum_circuit(THRESHOLD, data):
    binary = np.array(data > THRESHOLD, dtype=np.float32)
    circuit = [convert_to_circuit(x) for x in binary]
    tf_circuit = tfq.convert_to_tensor(circuit)

    return tf_circuit

# ---------------------------------------------------------------------------- #
## Build the model circuit of Quantum Neural Network
class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)

def create_quantum_model(index):
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
    # data_qubits = cirq.GridQubit.rect(8, 8)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()
    
    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))
    
    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

    # Then add layers (experiment by adding more).
    if index == 0:
        builder.add_layer(circuit, cirq.YY, "yy1")
        builder.add_layer(circuit, cirq.YY, "yy2")
    elif index == 1:
        builder.add_layer(circuit, cirq.YY, "yy3")
    elif index == 2:
        builder.add_layer(circuit, cirq.XX, "xx2")
    elif index == 3:
        builder.add_layer(circuit, cirq.ZZ, "XX1")
    elif index == 4:
        builder.add_layer(circuit, cirq.YY, "zz1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)


def load_qnn_model(index):
    model_circuit, model_readout = create_quantum_model(index)
    model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        tfq.layers.PQC(model_circuit, model_readout),
    ])

    return model


# ---------------------------------------------------------------------------- #
## Hinge Accuracy
def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)

# ---------------------------------------------------------------------------- #
## Training the QNN model:
def compile_qnn_model(training_data_path, training_data_labels_path, testing_data_path, testing_data_labels_path, THRESHOLD, validate,index):
    print("Compiling model for index = ", index)
    x_train, y_train, x_test, y_test = load_dataset(training_data_path, training_data_labels_path, testing_data_path, testing_data_labels_path, index)
    model = load_qnn_model(index)
    model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[hinge_accuracy])

    print("\n\nQNN Model Summary : ")
    print(model.summary())

    # Quantum Circuits
    x_train_tfcirc = quantum_circuit(THRESHOLD, x_train)
    x_test_tfcirc = quantum_circuit(THRESHOLD, x_test)
    y_train_hinge = 2.0 * y_train - 1.0
    y_test_hinge = 2.0 * y_test - 1.0

    # Training the model
    EPOCHS = 15
    BATCH_SIZE = 32
    NUM_EXAMPLES = len(x_train_tfcirc)

    x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
    y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

    print("\n\nTASK: Training the QNN model :")
    qnn_history = model.fit(
        x_train_tfcirc_sub, y_train_hinge_sub,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test_tfcirc, y_test_hinge))
    
    # Saving the model
    print("\nTASK: Saving the QNN model :\n")
    model.save_weights("src/backend/utils/model_store/")
    print("QNN Model Saved on your local machine !!!")
    
    if(validate):
        print("\n\nEvaluating the QNN model on the dataset :")
        qnn_results = model.evaluate(x_test_tfcirc, y_test)
        qnn_results2 = model.evaluate(x_train_tfcirc, y_train)
        y_pred = model.predict(x_test_tfcirc)
        y_pred_check = []
        for i in range(len(y_pred)):
            grade = y_pred[i][0]
            val = 1 if(grade >0) else 0
            y_pred_check.append(val)

        print(y_test - y_pred_check)
        print(f"QNN Model Accuracy on Test Data: {round(qnn_results[1] * 100, 4)}%")
        print(f"QNN Model Accuracy on Training Data: {round(qnn_results2[1] * 100, 4)}%")

    return model

# ---------------------------------------------------------------------------- #
## Classify the input image
# def classify(THRESHOLD, image_path, model0, model1, model2, model3, model4):
#     # print("TASK : Loading the Image into the classifier")
#     raw_image = tf.keras.utils.load_img(image_path, target_size=(4,4), color_mode = 'grayscale')
#     image_data = np.array(raw_image)/255.0
#     tf_circuit = quantum_circuit(THRESHOLD, image_data[0:1])
#     print(tf_circuit)
#     print(image_data)
#     response0 = model0.predict(tf_circuit)[0,0]
#     response1 = model1.predict(tf_circuit)[0,0]
#     response2 = model2.predict(tf_circuit)[0,0]
#     response3 = model3.predict(tf_circuit)[0,0]
#     response4 = model4.predict(tf_circuit)[0,0]
#     print(image_path)
#     print("Response 0 = ", response0)
#     print("Response 1 = ", response1)
#     print("Response 2 = ", response2)
#     print("Response 3 = ", response3)
#     print("Response 4 = ", response4)

#     print("Response = ", model0.predict(tf_circuit) )

#     a=(response0, response1, response2, response3, response4)
#     response_check = max(a)

#     print(response_check)
#     print()

#     if(response_check == response0):
#         return 0
#     elif(response_check == response1):
#         return 1
#     elif(response_check == response2):
#         return 2
#     elif(response_check == response3):
#         return 3
#     else:
#         return 4
    


def classify(THRESHOLD, image_number, model0, model1, model2, model3, model4):
    # print("TASK : Loading the Image into the classifier")
    # raw_image = tf.keras.utils.load_img(image_path, target_size=(4,4), color_mode = 'grayscale')
    testing_data = pd.read_csv("src/dataset/testing_data/testing_data.csv")
    image_data = testing_data[image_number : image_number+1]
    # print(image_data)
    tf_circuit = quantum_circuit(THRESHOLD, image_data[0:1])
    # print(tf_circuit)
    # print(image_data)
    response0 = model0.predict(tf_circuit)[0,0]
    response1 = model1.predict(tf_circuit)[0,0]
    response2 = model2.predict(tf_circuit)[0,0]
    response3 = model3.predict(tf_circuit)[0,0]
    response4 = model4.predict(tf_circuit)[0,0]
    
    # print(image_path)
    # print("Response = ", model0.predict(tf_circuit))

    print(f"Predicted Values :\nModel-0 : {response0}\nModel-1 : {response1}\nModel-2 : {response2}\nModel-3 : {response3}\nModel-4 : {response4}\n")


    a=(response0, response1, response2, response3, response4)
    response_check = max(a)

    # print(response_check)
    # print()

    if(response_check == response0):
        return 0
    elif(response_check == response1):
        return 1
    elif(response_check == response2):
        return 2
    elif(response_check == response3):
        return 3
    else:
        return 4
    

# ---------------------------------------------------------------------------- #

