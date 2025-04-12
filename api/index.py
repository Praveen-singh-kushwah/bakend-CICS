from mangum import Mangum
from main import app  # Import your FastAPI app from main.py

handler = Mangum(app, lifespan="off")