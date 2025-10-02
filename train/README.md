#### Установка на RTX 3080
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
 
python3 -m venv venv
source venv/bin/activate
 
# PyTorch с CUDA 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# HuggingFace экосистема
pip install transformers datasets accelerate
  
# Метрики
pip install seqeval

# Утилиты
pip install pandas numpy
 
или
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
 
# Запуск
python train.py


### Для конвертирования в ONXX
1. Создаём проект и окружение
   python3 -m venv onnx-env
   source onnx-env/bin/activate

2. Зависимости
   pip install -r requirements_onxx.txt


# Запуск
python export_onnx.py 