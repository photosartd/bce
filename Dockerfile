FROM python:3.9
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip --no-cache-dir install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
RUN pip --no-cache-dir install torch-geometric-temporal==0.54.0

COPY . .
WORKDIR .
RUN pip install -e .
ARG command=torch_bce/main.py
ENV command_env=$command
#ENV PYTHONPATH "${PYTHONPATH}:/"
ENTRYPOINT python ${command_env}
#CMD "python ${command}"