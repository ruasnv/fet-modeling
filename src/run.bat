@echo off
echo --- Running FET Pipeline ---

echo 1. Running EDA...
python main.py --mode eda

echo 2. Running GAN Augmentation...
python main.py --mode gan

echo 3. Running Cross Validation (SimpleNN)...
python main.py --mode cv --model simplenn

echo 4. Training Final Model...
python main.py --mode train --model simplenn

echo Done! Check the 'results' folder.
pause