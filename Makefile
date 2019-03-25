all:

build:
	docker build -t rc .

carma1: build
	mkdir -p experiments
	docker run --rm --user $$(id -u) -it -v $(PWD)/experiments:$(PWD)/experiments -w $(PWD)/experiments  rc carma1

carma-compute-equilibria: build
	mkdir -p experiments
	docker run --rm --user $$(id -u) -it -v $(PWD)/experiments:$(PWD)/experiments -w $(PWD)/experiments  rc carma-compute-equilibria
