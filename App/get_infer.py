import torch
from PIL import Image
from contextlib import nullcontext
from torchvision import transforms
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


async def get_device():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

async def predict_with_model(text, model, tokenizer, model_type="torch", max_length=512, device="cpu"):
    # Split the text into paragraphs based on double newlines or line breaks
    paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]

    chunks_text, chunks_labels = [], []

    if model_type == "torch":
        model.eval()

    with torch.no_grad() if model_type == "torch" else nullcontext():
        for paragraph in paragraphs:
            # Tokenize and prepare input for the model
            inputs = tokenizer(paragraph, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
            
            if model_type == "torch":
                inputs = inputs.to(device)
                outputs = model(**inputs)
                pred_label = torch.argmax(outputs.logits, dim=1).item()
            elif model_type == "onnx":
                onnx_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
                outputs = model.run(None, onnx_inputs)
                pred_label = int(outputs[0].argmax())

            chunks_text.append(paragraph)
            chunks_labels.append(pred_label)
    
    label_map = {
        1: "AI",
        0: "H"
    }

    return chunks_text, [label_map[label] for label in chunks_labels]

async def predict_image(model, image_path, device, idx_to_class):

    image = test_transforms(image_path).unsqueeze(0)  # Add batch dimension

    # Run inference
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)
    
    print(f"idx: {idx_to_class}")
    print(f"Prediction: {prediction}")
    # Map prediction to class
    predicted_class = idx_to_class.get(prediction.item(), "Unknown")
    print(f"Predicted class: {predicted_class}")
    return predicted_class