VS = code
INSTALL = --install-extension

all: 
	$(VS) $(INSTALL) ms-python.python
	$(VS) $(INSTALL) ms-python.vscode-pylance
	$(VS) $(INSTALL) ms-toolsai.jupyter
	$(VS) $(INSTALL) ms-toolsai.jupyter-keymap
	$(VS) $(INSTALL) ms-toolsai.jupyter-renderers