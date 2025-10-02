from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer

model_path = "model"
onnx_path = "onxx_name"

# Экспорт в ONNX
main_export(
    model_name_or_path=model_path,
    output=onnx_path,
    task="token-classification",
    opset=17,
    device="cpu"
)

# Сохраняем токенизатор
AutoTokenizer.from_pretrained(model_path).save_pretrained(onnx_path)

print("Экспорт через main_export завершён.")
