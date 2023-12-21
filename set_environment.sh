#/bin/bash

# Create a new environment: glen
conda create --name glen python=3.8 --yes

# Activate the environment
conda activate glen

# Install the requirements
pip install -r requirements.txt

# Install the package
pip install --editable .

# Install torch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge --yes


# Install GradCache
cd GradCache
pip install -e .
cd ..

# Test
python -c "import tevatron; print(tevatron.__version__)"



