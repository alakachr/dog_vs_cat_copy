from fastapi.testclient import TestClient
import sys

sys.path.append("..")
from main import app

client = TestClient(app)
filename = "cat.37.jpg"


def test_read_main():
    """Unit testing of get request"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome from the API"}


def test_read_prediction():
    """Unit testing of post  request"""
    response = client.post(
        "/predict",
        files={"file": ("filename", open(filename, "rb"), filename[:-4] + "/jpeg")},
    )
    assert response.status_code == 200
    assert response.json() is not None


test_read_main()
test_read_prediction()

print("Testing API Ended")
