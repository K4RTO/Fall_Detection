import copy
import torch
from torchvision import transforms


def result_generate(model_path, data):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path, map_location=device)
    # Set model to evaluation mode
    model.eval()

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize to the same size used in training
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    # 遍历目录中的所有图片文件
    results = []
    result = []
    count = 0

    # Iterate through images in the directory
    for audio_slice_spectrogram_image in data:
        image = audio_slice_spectrogram_image.convert('RGB')
        image = transform(image).unsqueeze(0)  # Add a batch dimension
        image = image.to(device)  # Move input to the same device as the model

        # Perform prediction
        with torch.no_grad():
            output = model(image)

        # Get predicted class
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        if predicted_class == 0:
            result.append(count)
            result.append(count+3)
            results.append(copy.deepcopy(result))
            result.clear()

        count += 1

    print(f"fall detected in the following time scale {results} (second)")
    # 返回结果映射
    return results
