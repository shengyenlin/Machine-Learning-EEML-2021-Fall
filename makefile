VS = code
INSTALL = --install-extension

all: install_python install_HW_packages

install_python:
	$(VS) $(INSTALL) "ms-python.python"
	$(VS) $(INSTALL) "ms-python.vscode-pylance"
	$(VS) $(INSTALL) "ms-toolsai.jupyter"
	$(VS) $(INSTALL) "ms-toolsai.jupyter-keymap"
	$(VS) $(INSTALL) "ms-toolsai.jupyter-renderers"

install_HW_packages:
	pip install -r requirements_HW3.txt
	pip install opencv-python
	pip install optuna
	pip install "notebook>=5.3" "ipywidgets>=7.5"
	pip install torchviz