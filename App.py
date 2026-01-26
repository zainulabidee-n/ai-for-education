import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# -----------------------
# Load Model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 10)
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 10)
model.to(device)
model.eval()

# -----------------------
# Image Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -----------------------
# Prediction Function
# -----------------------
def predict(image):
    image = Image.fromarray(image).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, 1).item()

    return f"Predicted Class: {pred}"

# -----------------------
# Gradio UI
# -----------------------
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Fashion MNIST Classifier"
).launch()