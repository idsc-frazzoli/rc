FROM python:3.7

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

RUN python setup.py develop --no-nodeps

CMD ["carma1"]
