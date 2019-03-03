

# Native 

## Install 

Install using:

    pip install -r requirements.txt
    python setup.py develop --no-deps


## Run

Run using:

    carma1
    
    
# Docker

## Build

    docker build -t rc .
    
## Run

    docker run -it -v $PWD/out-experiments:/out-experiments -w / rc
    
