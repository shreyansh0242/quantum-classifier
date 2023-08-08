## Diabetic Retinotherapy Classifier - Backend
import pandas as pd
import numpy as np
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
            # model = classifier.compile_qnn_model(training_data_path, training_data_labels_path, testing_data_path, testing_data_labels_path, THRESHOLD, validate)
            model0 = classifier.compile_qnn_model(training_data_path, training_data_labels_path, testing_data_path, testing_data_labels_path, 0.1, validate, 0)
            model1 = classifier.compile_qnn_model(training_data_path, training_data_labels_path, testing_data_path, testing_data_labels_path, 0.1, validate, 1)
            model2 = classifier.compile_qnn_model(training_data_path, training_data_labels_path, testing_data_path, testing_data_labels_path, 0.1, validate, 2)
            model3 = classifier.compile_qnn_model(training_data_path, training_data_labels_path, testing_data_path, testing_data_labels_path, 0.1, validate, 3)
            model4 = classifier.compile_qnn_model(training_data_path, training_data_labels_path, testing_data_path, testing_data_labels_path, 0.1, validate, 4)
        except Exception as e:
            print(f"ERROR : {e}")

    else:
        # model = classifier.load_qnn_model()
        # model.load_weights("src/backend/utils/model_store/")
        pass

    # Evaluate the input image
    # image_path = "src/dataset/testing_data/images/IDRiD_001.jpg"
    # print("\nTASK : Loaded the QNN Classifier model")
    # print("TASK : Evaluating the input image1 :")
    # response = classifier.classify(0.1, image_path, model0, model1, model2, model3, model4)
    # print(f"DR Possibility for input image : {response*1} ")
    
    #--------------------------------------------------------------------#
    # y_pred = []
    # for image_name in os.listdir(testing_data_path):
    #     image_path = f"{testing_data_path}{image_name}"
    #     response = classifier.classify(0.1, image_path, model0, model1, model2, model3, model4)
    #     y_pred.append(response)

    # y_test = []
    # testing_data_labels = pd.read_csv(testing_data_labels_path)
    # for i in range(len(testing_data_labels)):
    #     grade = testing_data_labels['retinopathy_grade'][i]
    #     val = grade
    #     y_test.append(val)
    
    # print("Printing test results: ")
    # print(y_pred)
    # print(np.array(y_test)-np.array(y_pred))


    y_test = []
    testing_data_labels = pd.read_csv(testing_data_labels_path)
    for i in range(len(testing_data_labels)):
        grade = testing_data_labels['retinopathy_grade'][i]
        val = grade
        y_test.append(val)

    y_pred = []
    # n = 103
    n = 5
    for i in range(n):
        print(f"Image-{i}", end = " ")
        response = classifier.classify(0.5, i, model0, model1, model2, model3, model4)
        print(f"Actual Value: {y_test[i]}")
        y_pred.append(response)
    
    print(y_pred)
    print(np.array(y_test)-np.array(y_pred))

