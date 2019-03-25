all:

build:
	docker build -t rc .

carma1: build
	docker run --rm -it -v $(PWD)/out-experiments:/out-experiments -w / rc carma1

carma-compute-equilibria: build
	docker run --rm -it -v $(PWD)/out-iterative:/out-iterative -w / rc carma-compute-equilibria

