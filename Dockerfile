FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# By default the container runs the evaluator's unit tests.
# To score your own predictions, mount a data folder and override the command (see README).
CMD ["python", "-m", "pytest", "-q", "evaluation"]
