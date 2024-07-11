import torch
import convert_onnx
from scripts.crack_module import Classifier

def main():
    model = Classifier(n_classes=2)
    model.load_state_dict(torch.load(""))
    model.eval()

    dummy_input = torch.randn(1, 3, 299, 299)


if __name__ == "__main__":
    main()