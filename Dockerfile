FROM python:3.12-slim
WORKDIR /usr/local/sudoku_solver_app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

EXPOSE 5000

CMD ["python3", "-m", "flask", "-A" ,"web_app/app.py", "run",  "--host=0.0.0.0"]