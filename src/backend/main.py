## Diabetic Retinotherapy Classifier - Backend
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import uvicorn
from utils import classifier, preprocessing

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded images
UPLOAD_DIR = os.environ["UPLOAD_DIR"]


# Create the upload directory if it doesn't exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


def train_models():
    training_data_path = "src/dataset/training_data/images/"
    training_data_labels_path = "src/dataset/training_data/training_labels.csv"
    testing_data_path = "src/dataset/testing_data/images/"
    testing_data_labels_path = "src/dataset/testing_data/testing_labels.csv"
    validate = False
    model_grade_0 = classifier.compile_qnn_model(training_data_labels_path, testing_data_labels_path, 0.1, validate, 0)
    model_grade_1 = classifier.compile_qnn_model(training_data_labels_path, testing_data_labels_path, 0.1, validate, 1)
    model_grade_2 = classifier.compile_qnn_model(training_data_labels_path, testing_data_labels_path, 0.1, validate, 2)
    model_grade_3 = classifier.compile_qnn_model(training_data_labels_path, testing_data_labels_path, 0.1, validate, 3)
    model_grade_4 = classifier.compile_qnn_model(training_data_labels_path, testing_data_labels_path, 0.1, validate, 4)
    print("DONE : Successfully Trained all the models\n\n")
    return model_grade_0, model_grade_1, model_grade_2, model_grade_3, model_grade_4

    



@app.post("/upload/")
async def upload_image(file: UploadFile):
    try:
        file_extension = file.filename.split(".")[-1]
        file_name = f"uploaded_image.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        with open(file_path, "wb") as image_file:
            image_file.write(file.file.read())

        img = cv2.imread(file_path)
        processed_image = preprocessing.preprocess_image(img)
        grade_dict = classifier.classify(THRESHOLD, processed_image, model_grade_0, model_grade_1, model_grade_2, model_grade_3, model_grade_4)
        predicted_grade = 'grade_0'
        predicted_value = grade_dict[predicted_grade]
        print('new response',predicted_value)

        for key in ['grade_1', 'grade_2', 'grade_3', 'grade_4']:
            value = grade_dict[key]
            if  value > predicted_value:
                predicted_value = value
                predicted_grade = key

        response_dict = {
            "model_grades": grade_dict,
            "predicted_grade": str(predicted_grade),
            "predicted_value": str(predicted_value),
            "message": "Successfully predicted the Diabetic Retinopathy Grade !!!"
            }
        return JSONResponse(content=response_dict, status_code=200)

    except Exception as err:
        print(f"ERROR : {err}")
        return JSONResponse(content={"message": "An error occurred"}, status_code=500)


if __name__ == "__main__":
    print("\n\n<---Indian Diabetic Retinopathy Classification using Quantum Neural Networks--->\n")
    model_grade_0, model_grade_1, model_grade_2, model_grade_3, model_grade_4 = train_models()
    THRESHOLD = 0.5
    uvicorn.run(app, port=8000)
