import streamlit as st
from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models

# -----------------------------
# Title & description
# -----------------------------
st.title(" Malaria Detection App (CNN + ResNet50)")
st.write("Upload a blood cell image to classify whether it is **Parasitized** or **Uninfected**.")

# -----------------------------
# Class names
# -----------------------------
class_names = ["Parasitized", "Uninfected"]

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Build the SAME model as in training
# -----------------------------
# No pretrained weights needed here, we load our own trained weights
base_model = models.resnet50(weights=None)
in_features = base_model.fc.in_features
base_model.fc = nn.Linear(in_features, 2)

# Load trained weights
STATE_PATH = "malaria_resnet50_best.pth"

try:
    state_dict = torch.load(STATE_PATH, map_location=device)
    base_model.load_state_dict(state_dict)
except FileNotFoundError:
    st.error(
        f"Model weights file '{STATE_PATH}' not found. "
        "Make sure it is in the same folder as this script."
    )
    st.stop()

model = base_model.to(device)
model.eval()

# -----------------------------
# Preprocessing
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def load_input_image(pil_img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalized tensor with batch dimension."""
    img = pil_img.convert("RGB")
    tensor = preprocess(img).unsqueeze(0)  # [1, C, H, W]
    return tensor.to(device)

# -----------------------------
# Streamlit UI
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a cell image (.png, .jpg, .jpeg)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)

    if st.button("Predict"):
        # Preprocess and predict
        with torch.no_grad():
            x = load_input_image(image)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs))
            pred_class = class_names[pred_idx]
            confidence = float(probs[pred_idx])

        st.subheader("Prediction")
        st.write(f"**Class:** {pred_class}")
        st.write(f"**Confidence:** {confidence:.3f}")
