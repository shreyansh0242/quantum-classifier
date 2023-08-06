## Diabetic Retinotherapy Classifier - Backend
import os
from utils import classifier

if __name__ == "__main__":

    # Global Variables
    training_data_path = "src/dataset/training_data/images/"
    training_data_labels_path = "src/dataset/training_data/training_labels.csv"
    testing_data_path = "src/dataset/testing_data/images/"
    testing_data_labels_path = "src/dataset/testing_data/testing_labels.csv"
    THRESHOLD = 0.5
    create_model = False
    validate = False

    print("\n\n<---Indian Diabetic Retinopathy Classification using Quantum Neural Networks--->\n")

    # if(len(os.listdir("src/backend/utils/model_store")) == 0):
    #     print("TASK : QNN model doesn't exists, Creating new one !!!")
    #     create_model = True

    create_model = True
    if(create_model):
        try:
            model = classifier.compile_qnn_model(training_data_path, training_data_labels_path, testing_data_path, testing_data_labels_path, THRESHOLD, validate)
        except Exception as e:
            print(f"ERROR : {e}")

    else:
        model = classifier.load_qnn_model()
        model.load_weights("src/backend/utils/model_store/")

    # Evaluate the input image
    image_path = "src/dataset/training_data/images/IDRiD_003.jpg"
    print("\nTASK : Loaded the QNN Classifier model")
    print("TASK : Evaluating the input image :")
    response = classifier.classify(THRESHOLD, image_path, model)
    print(f"DR Possibility for input image : {response*100} % ")

