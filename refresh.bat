pip uninstall cbas-headless -y
python setup.py bdist_wheel
pip install .\dist\cbas_headless-0.1.0-py3-none-any.whl