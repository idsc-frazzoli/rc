

# Native 

## Install 

Install using:

    pip install -r requirements.txt
    python setup.py develop --no-deps


## Run

Run using:

    carma1
    
**new**: Run the Nash equilibria finder using:

    python3 -m carma.iterative    
    
    
# Docker

## Build

    docker build -t rc .
    
## Run

    docker run -it -v $PWD/out-experiments:/out-experiments -w / rc
    

QmY2Putz3yVXBpDt68WYUUBajQrb8Gn1mR1gHWoD5niKtX