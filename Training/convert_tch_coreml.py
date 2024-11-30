import torch
import coremltools as ct
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # Return logits directly

def convert_pytorch_to_coreml(model_name):
    # Load your trained model
    model_path = f'./results_{model_name}/checkpoint-last'
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode

    # Wrap the model to return logits directly
    wrapped_model = WrappedModel(model)

    # Load the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Prepare a sample input for tracing
    sample_text = "This is a sample text for model conversion."
    inputs = tokenizer(sample_text, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs['input_ids'].to(torch.int32)  # Cast to int32
    attention_mask = inputs['attention_mask'].to(torch.int32)  # Cast to int32

    # Trace the wrapped model with sample inputs
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapped_model,
            (input_ids, attention_mask),
            strict=False  # Allow non-Tensor inputs
        )

    # Define finite ranges for input shapes
    batch_size_range = ct.RangeDim(lower_bound=1, upper_bound=1)  # Fixed batch size of 1
    sequence_length_range = ct.RangeDim(lower_bound=1, upper_bound=512)  # Sequence length up to 512 tokens

    input_ids_shape = ct.Shape(shape=(batch_size_range, sequence_length_range))
    attention_mask_shape = ct.Shape(shape=(batch_size_range, sequence_length_range))

    # Convert the traced model to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=input_ids_shape, dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=attention_mask_shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits")
        ],
        compute_units=ct.ComputeUnit.ALL,  # Use ALL to leverage CPU and GPU
        minimum_deployment_target=ct.target.iOS15  # Set minimum deployment target explicitly
    )

    # Save the CoreML model
    mlmodel_path = f'./results_{model_name}/model.mlpackage'
    mlmodel.save(mlmodel_path)
    print(f"CoreML model saved at {mlmodel_path}")

# Convert all three models
if __name__ == "__main__":
    model_names = [
        "whole_article_model",
        # "paragraph_model",
        # "sentence_model"
    ]
    for model_name in model_names:
        convert_pytorch_to_coreml(model_name)
