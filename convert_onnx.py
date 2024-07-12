import torch
import onnx
from scripts.crack_module import Classifier
from scripts.inception import InceptionCustom

def main():
    num_classes = 2
    inception_model = InceptionCustom(num_classes=num_classes)
    model = Classifier(net=inception_model, lr=0.001)
    model.load_state_dict(torch.load("./model/inception_model_crack.pth"))
    model.eval()

    dummy_input = torch.randn(1, 3, 299, 299)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            "./Crack_inception.onnx",
            input_names=["input"],
            output_names=["output"],
            verbose=True,
        )

if __name__ == "__main__":
    main()