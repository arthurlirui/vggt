import torch
from vggt.models.vggt import VGGT


def quantize_vggt(model_path, quantized_model_path):
    # 加载预训练模型
    model = VGGT.from_pretrained(model_path).to("cpu")
    model.eval()

    # 动态量化配置
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},  # 指定需要量化的层类型
        dtype=torch.qint8,  # 目标数据类型
        qconfig_spec={
            torch.nn.Linear: torch.quantization.default_dynamic_qconfig,
            torch.nn.Conv2d: torch.quantization.default_dynamic_qconfig
        }
    )

    # 保存量化模型
    torch.save(quantized_model.state_dict(), quantized_model_path)
    return quantized_model


def load_quantized_model(model_path):
    model = VGGT()
    quantized_state_dict = torch.load(model_path)
    model.load_state_dict(quantized_state_dict)
    model.eval()
    # 设置量化引擎
    model = torch.quantization.prepare(model, inplace=False)
    model = torch.quantization.convert(model, inplace=False)
    return model.to("cpu")  # 量化模型在CPU上通常表现更好


if __name__ == "__main__":
    # 使用示例
    quantized_model = quantize_vggt("facebook/VGGT-1B", "vggt_int8.pt")
    # 加载量化模型并推理
    model = load_quantized_model("vggt_int8.pt")
    image_names = ["examples/kitchen/images/00.png", "examples/kitchen/images/01.png"]
    images = load_and_preprocess_images(image_names)

    with torch.no_grad():
        predictions = model(images)  # 使用INT8量化模型推理
