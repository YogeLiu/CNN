import torch
from models.alexnet import AlexNet
from PIL import Image
from utils.transforms import get_test_transform
import config
import os

def predict_image(image_path, model_path):
    # 加载模型
    model = AlexNet(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model = model.to(config.DEVICE)
    model.eval()
    
    # 加载和预处理图片
    image = Image.open(image_path).convert('RGB')
    transform = get_test_transform()
    image = transform(image).unsqueeze(0).to(config.DEVICE)
    
    # 预测
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
    return 'cat' if predicted.item() == 0 else 'dog'

if __name__ == '__main__':
    import pandas as pd 
    from concurrent.futures import ThreadPoolExecutor
    count = 0
    # Create DataFrame
    file = pd.DataFrame(columns=['id', 'label'])
    dir = 'data/test'
    file_paths = []

    # Collect image paths
    for img in os.listdir(dir):
        file_paths.append(os.path.join(dir, img))
    file_paths.sort()

    # Process images and save results concurrently
    def process_image(file_path):
        result = predict_image(file_path, 'saved_models/model_epoch_50.pth')
        return file_path, result

    with ThreadPoolExecutor(max_workers=10) as executor:  # Set the number of concurrent processes
        results = list(executor.map(process_image, file_paths))

    # Append rows to DataFrame
    for file_path, result in results:
        file = pd.concat([file, pd.DataFrame([[file_path, result]], columns=['id', 'label'])], ignore_index=True)

    # Save the result to CSV
    file.to_csv('result.csv', index=False)