all:

build:
	docker build -t rc .

carma1: build
	mkdir -p experiments
	docker run --rm --user $$(id -u) -it -v $(PWD)/experiments:$(PWD)/experiments -w $(PWD)/experiments  rc carma1

collect: build
	docker run --rm --user $$(id -u) -it -v $(PWD)/experiments:$(PWD)/experiments -w $(PWD)/experiments  rc python3 -m carma.collect_results

carma-compute-equilibria: build
	mkdir -p experiments
	docker run --rm --user $$(id -u) -it -v $(PWD)/experiments:$(PWD)/experiments -w $(PWD)/experiments  rc carma-compute-equilibria all

carma-overview: build
		mkdir -p experiments
		docker run --rm --user $$(id -u) -it -v $(PWD)/experiments:$(PWD)/experiments -w $(PWD)/experiments  rc carma-overview
