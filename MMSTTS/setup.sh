#!/bin/bash

# Clone the GitHub Repository
git clone https://github.com/jaywalnut310/vits.git

# Navigate into the cloned repository and install required packages
cd vits/
pip install Cython==0.29.21 librosa==0.8.0 phonemizer==2.2.1 scipy numpy torch torchvision matplotlib Unidecode==1.1.1

# Compile the Monotonic Align Extension
cd monotonic_align/
mkdir -p monotonic_align
python3 setup.py build_ext --inplace
cd ..

# Return to the root directory
cd ..

echo "Setup complete!"