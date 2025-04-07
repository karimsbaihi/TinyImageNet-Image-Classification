# import torch
# import pandas as pd
# from model import create_model  # Importing the model function
# from utils import get_dataloaders  # Importing the dataloader function

# def generate_submission(model_checkpoint, test_dir):
#     # Load trained model
#     model = create_model()
#     model.load_state_dict(torch.load(model_checkpoint))
    
#     # Process test images (no labels)
#     test_images = get_test_images(test_dir)
#     predictions = []

#     for image in test_images:
#         output = model(image)
#         _, predicted_class = torch.max(output, 1)
#         predictions.append(predicted_class.item())
    
#     # Save predictions to CSV
#     with open('experiments/results/submission.csv', 'w') as f:
#         f.write("id,class\n")
#         for idx, prediction in enumerate(predictions):
#             f.write(f"{idx},{prediction}\n")
