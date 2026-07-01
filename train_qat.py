"""
Quantization-Aware Training (QAT) for the ZedBoard MNIST accelerator.

Standalone equivalent of mnist_quantized.ipynb, with the *downsized* architecture
that fits the Zynq-7020 (XC7Z020) on the ZedBoard:

    conv1 : 1  -> 16   (4-bit weights)
    conv2 : 16 -> 16   (2-bit weights)
    fc1   : 144 -> 64  (2-bit weights)      (144 = 16 * 3 * 3 after 4x4 pool)
    fc2   : 64 -> 10   (4-bit weights)
    activations: 8-bit input, 2-bit ReLUs

This is the model that reached ~97.5% test accuracy and fit at ~34k LUT / 43 DSP.
Trains with Brevitas (fake-quant in the loop), saves the PyTorch weights, exports
a FINN-ready QONNX model, and prints test accuracy.

Outputs (written next to this script):
    mnist_cnn.pth           - trained weights
    mnist_finn_ready.onnx   - QONNX export -> feed into hls4ml (build3.ipynb)

Run:
    python train_qat.py
Requires: torch, torchvision, brevitas   (GPU used automatically if available)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat   # signed, for weights
from brevitas.quant import Int8ActPerTensorFloat      # signed, for the input
from brevitas.quant import Uint8ActPerTensorFloat     # unsigned, for ReLU outputs
from brevitas.quant import Int8Bias
from brevitas.export import export_qonnx

# anchor all relative paths to this script's folder so it runs from anywhere
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---- hyperparameters ------------------------------------------------------
NUM_EPOCHS = 10
BATCH_SIZE = 64
LR         = 5e-4
PTH_FILE   = "mnist_cnn.pth"
ONNX_FILE  = "mnist_finn_ready.onnx"


# ---- model ----------------------------------------------------------------
class QMNIST(nn.Module):
    """Downsized quantized CNN sized to fit the ZedBoard's XC7Z020.

    The MaxPool / Dense LUT cost scales with channel count, so conv2 is kept at
    16 filters (cascades into pool2 and fc1). Bit-widths match a 2-bit-activation
    / 2-4-bit-weight budget so the fixed-point hardware stays small.
    """

    def __init__(self):
        super().__init__()
        # 8-bit signed input quant (raw pixels -- keep resolution here)
        self.inp_quant = qnn.QuantIdentity(
            bit_width=8, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

        # conv1: 1 -> 16 filters, 4-bit weights (first layer touches raw input)
        self.conv1 = qnn.QuantConv2d(
            1, 16, kernel_size=3, padding=1,
            weight_quant=Int8WeightPerTensorFloat, weight_bit_width=4,
            bias=True, bias_quant=Int8Bias)
        self.relu1 = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat, bit_width=2, return_quant_tensor=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)     # 28x28 -> 14x14

        # conv2: 16 -> 16 filters, 2-bit weights
        self.conv2 = qnn.QuantConv2d(
            16, 16, kernel_size=3, padding=1,
            weight_quant=Int8WeightPerTensorFloat, weight_bit_width=2,
            bias=True, bias_quant=Int8Bias)
        self.relu2 = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat, bit_width=2, return_quant_tensor=True)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)    # 14x14 -> 3x3

        # fc1: 16*3*3 = 144 -> 64, 2-bit weights
        self.fc1 = qnn.QuantLinear(
            16 * 3 * 3, 64,
            weight_quant=Int8WeightPerTensorFloat, weight_bit_width=2,
            bias=True, bias_quant=Int8Bias)
        self.relu3 = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat, bit_width=2, return_quant_tensor=True)

        # fc2: 64 -> 10 logits, 4-bit weights (no ReLU -- logits go negative)
        self.fc2 = qnn.QuantLinear(
            64, 10,
            weight_quant=Int8WeightPerTensorFloat, weight_bit_width=4,
            bias=True, bias_quant=Int8Bias)

    def forward(self, x):
        x = self.inp_quant(x)
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    # MNIST preprocessing: ToTensor (0..1) + Normalize(mean, std).
    # The bare-metal C app must replicate this exact normalization.
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081]),
    ])

    train_data = datasets.MNIST(root="./data", train=True,  download=True, transform=data_transforms)
    test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=data_transforms)

    # num_workers=0 -> avoids the Windows DataLoader spawn crash
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_data,  batch_size=1000,       shuffle=False, num_workers=0)

    # Brevitas uses ops MPS doesn't support -> CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = QMNIST().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- train ----
    for epoch in range(NUM_EPOCHS):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {running / len(train_loader):.10f}")

    torch.save(model.state_dict(), PTH_FILE)
    print("Saved weights ->", PTH_FILE)

    # ---- evaluate ----
    model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, pred = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # ---- export QONNX for hls4ml ----
    model.cpu().eval()   # export must trace on CPU
    export_qonnx(model, torch.randn(1, 1, 28, 28), ONNX_FILE)
    print("Exported ->", ONNX_FILE)


if __name__ == "__main__":
    main()
