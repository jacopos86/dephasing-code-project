build :
	pip install -r requirements.txt
install :
	python setup.py install
.PHONY :
	clean
clean :
	rm -rf ./pydephasing/*~ ./pydephasing/__pycache__ ./build/lib/pydephasing/*
