import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
import os
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)

# Helper to create a fake valid image
def create_dummy_image():
    file = BytesIO()
    Image.new('RGB', (100, 100), color='green').save(file, format='JPEG')
    file.seek(0)
    return file

@pytest.fixture
def mock_token(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-secret-token")

def test_health_check_does_not_load_model():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_unauthorized():
    response = client.post("/predict", files={"file": ("test.jpg", create_dummy_image(), "image/jpeg")})
    assert response.status_code == 401

@patch("app.ai_service.get_model")
def test_predict_valid_image(mock_get_model):
    # Mock the model so we don't actually load YOLO during tests
    mock_model = MagicMock()
    mock_model.names = {0: "Test_Disease"}
    
    # Create a fake result object
    mock_box = MagicMock()
    mock_box.cls = [0]
    mock_box.conf = [0.85]
    mock_box.xyxy = [[10, 10, 50, 50]]
    
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.probs = None
    
    mock_model.predict.return_value = [mock_result]
    mock_get_model.return_value = mock_model

    # Send valid request with auth header
    response = client.post(
        "/predict", 
        headers={"Authorization": "Bearer test-secret-token"},
        files={"file": ("test.jpg", create_dummy_image(), "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["detections"][0]["class_name"] == "Test_Disease"
    assert data["detections"][0]["confidence"] == 0.85

def test_predict_invalid_image():
    # Send a text file instead of an image
    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer test-secret-token"},
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unable to read image" in response.json()["detail"]