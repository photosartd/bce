FROM python:3.9
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip --no-cache-dir install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

COPY . .
WORKDIR .
ENV PYTHONPATH "${PYTHONPATH}:/"

CMD ["python", "src/main.py"]