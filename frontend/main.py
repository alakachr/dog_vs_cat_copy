# code inspired by https://testdriven.io/blog/fastapi-streamlit/

import requests
import streamlit as st


st.set_option("deprecation.showfileUploaderEncoding", False)


st.title("Cat vs Dog classifier web app")

image = st.file_uploader("Choose an image")


# displays a button
if st.button("Predict"):
    if image is not None:  # Making sure user upload an image file
        files = {"file": image.getvalue()}

        res = requests.post("http://backend:8080/predict", files=files)

        prediction = res.json()
        label = prediction.get("prediction")

        st.text("It is a " + label + " !")
        st.image(image, width=500)
