FROM python:3.9

WORKDIR /workspaces

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip

COPY requirements.txt .

RUN python -m pip install -r requirements.txt

COPY . .

CMD ["python", "train.py"]