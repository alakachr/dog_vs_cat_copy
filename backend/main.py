# code inspired by https://testdriven.io/blog/fastapi-streamlit/


import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile

from PIL import Image

import inference


app = FastAPI(
    title="Cat vs Dogs",
    description="This API was built with FastAPI and exists to predict whether there is a dog or a cat in a picture",
)


@app.get(
    "/",
    name="Index",
    summary="Initialize the web app",
    tags=["Routes"],
    responses={
        200: {
            "content": {
                "application/json": {"example": {"message": "Welcome from the API"}}
            },
        }
    },
)
def read_root():
    """Initialize the web app"""
    return {"message": "Welcome from the API"}


@app.post(
    "/predict",
    name="Predict",
    summary="Returns the predicted label",
    tags=["Routes"],
    response_description="Successful Response: image label predicted by the model",
    responses={
        200: {
            "content": {"application/json": {"example": {"prediction": "cat"}}},
        }
    },
)
def get_image(file: UploadFile = File(...)):
    """Load the input image and return its predicted label.
    This function is called when the user clicks on the predict button.

    Arguments
    ----------
    file: the image file from the upload button

    Returns
    --------
    label: the predicted label
    """

    image = Image.open(file.file)

    pred = inference.inference(image)

    return {"prediction": pred}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
