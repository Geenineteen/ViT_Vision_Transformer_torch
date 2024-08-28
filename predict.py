"""
This script is used to load a trained Vision Transformer (ViT) model, calculate and display the prediction along with the confidence score.
"""
import os

import torch
from PIL import Image
import cv2

from dataloader import data_transform
from utils import create_model, model_parallel,remove_dir_and_create_dir
from config import args


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load image
    # image_path = "./path/to/test/image"
    image_path = "F:\Transformer\\torch_Vision_Transformer\dataset\dataset\daisy.jpg"
    assert os.path.exists(image_path), "file: '{}' dose not exist.".format(image_path)

    image = Image.open(image_path)

    # [N, C, H, W]
    image = data_transform["val"](image)
    # expand batch dimension
    image = torch.unsqueeze(image, dim=0)

    # create model
    model = create_model(args)
    model = model_parallel(args, model).to(device)
    # load model weights
    model_weight_path = "{}\weights\epoch=27_val_acc=0.9560.pth".format(args.summary_dir)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    model.eval()
    with torch.no_grad():
        # Predict class
        output = torch.squeeze(model(image.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        index = torch.argmax(predict).numpy()

    predict_label = args.label_name[index]
    predict_prob = predict[index].numpy()

    print("prediction: {}   prob: {:.3}\n".format(predict_label,
                                                predict_prob))
    for i in range(len(predict)):
        print("class: {}   prob: {:.3}".format(args.label_name[i],
                                               predict[i].numpy()))

    # Visualization
    img = cv2.imread(image_path)

    results_dir = args.summary_dir + "/results"
    remove_dir_and_create_dir(results_dir)

    cv2.putText(img, "pred: {}  prob: {:.3}".format(predict_label, predict_prob), (5, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 0, 0), 2)
    cv2.imshow("predict",img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(results_dir + '/predicted_'+ predict_label +'.jpg', img)

if __name__ == '__main__':
    main()
