pip install --upgrade pip

# install overcooked_ai
cd overcooked_ai
pip install -e .

cd ..
pip install -r requirements.txt

python setup.py develop
