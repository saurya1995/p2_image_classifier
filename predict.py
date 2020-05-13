import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
import argparse
import numpy as np
import json


def process_image(numpy_image):
    tensor_img = tf.image.convert_image_dtype(numpy_image, tf.float32)
    resized_img = tf.image.resize(numpy_image,(224,224)).numpy()
    return resized_img/255

def get_class_names(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    # Remapping as Class names have index starting from 1 to 102, whereas the datasets have label indices from 0 to 101
    class_new_names = dict()
    for key in class_names:
        class_new_names[str(int(key)-1)] = class_names[key]
    return class_new_names


def predict(image_path, model_path, top_k, all_class_names):
    top_k = int(top_k)
    print(top_k, type(top_k))
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})

    img = Image.open(image_path)
    test_image = np.asarray(img)

    # processing the image
    processed_test_image = process_image(test_image)
    
    # Predicting probabilities
    prob_preds = model.predict(np.expand_dims(processed_test_image,axis=0))
    prob_preds = prob_preds[0].tolist()

    # top k predictions
    values, indices= tf.math.top_k(prob_preds, k=top_k)
    probs_topk = values.numpy().tolist()
    print("top {} probs:".format(top_k),probs_topk)
    classes_topk = indices.numpy().tolist()
    class_labels = [all_class_names[str(i)] for i in classes_topk]
    print('top {} class labels:'.format(top_k), class_labels)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument("image_path",help="Image Path", default="")
    parser.add_argument("saved_model",help="Model Path", default="")
    parser.add_argument("--top_k", help="Fetch top k predictions", required = False, default = 3)
    parser.add_argument("--category_names", help="Class map json file", required = False, default = "label_map.json")
    args = parser.parse_args()

    all_class_names = get_class_names(args.category_names)

    predict(args.image_path, args.saved_model, args.top_k, all_class_names)