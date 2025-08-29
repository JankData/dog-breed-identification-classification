import os
import traceback
import numpy as np
from PIL import Image
import gradio as gr

import tensorflow as tf
import tensorflow_hub as hub

FULL_MODEL_PATH = "models/194834-27082025-full-resnetv2-Adam.keras"
FEATURE_VECTOR_DIR = "models/resnet-v2-tensorflow2-50-feature-vector-v2"
WEIGHTS_PATH = "models/resnetv2_classifier.weights.h5"
IMG_SIZE = 224

CLASSES = [
'affenpinscher','afghan_hound','african_hunting_dog','airedale','american_staffordshire_terrier',
'appenzeller','australian_terrier','basenji','basset','beagle',
'bedlington_terrier','bernese_mountain_dog','black-and-tan_coonhound','blenheim_spaniel','bloodhound',
'bluetick','border_collie','border_terrier','borzoi','boston_bull',
'bouvier_des_flandres','boxer','brabancon_griffon','briard','brittany_spaniel',
'bull_mastiff','cairn','cardigan','chesapeake_bay_retriever','chihuahua',
'chow','clumber','cocker_spaniel','collie','curly-coated_retriever',
'dandie_dinmont','dhole','dingo','doberman','english_foxhound',
'english_setter','english_springer','entlebucher','eskimo_dog','flat-coated_retriever',
'french_bulldog','german_shepherd','german_short-haired_pointer','giant_schnauzer','golden_retriever',
'gordon_setter','great_dane','great_pyrenees','greater_swiss_mountain_dog','groenendael',
'ibizan_hound','irish_setter','irish_terrier','irish_water_spaniel','irish_wolfhound',
'italian_greyhound','japanese_spaniel','keeshond','kelpie','kerry_blue_terrier',
'komondor','kuvasz','labrador_retriever','lakeland_terrier','leonberg',
'lhasa','malamute','malinois','maltese_dog','mexican_hairless',
'miniature_pinscher','miniature_poodle','miniature_schnauzer','newfoundland','norfolk_terrier',
'norwegian_elkhound','norwich_terrier','old_english_sheepdog','otterhound','papillon',
'pekinese','pembroke','pomeranian','pug','redbone',
'rhodesian_ridgeback','rottweiler','saint_bernard','saluki','samoyed',
'schipperke','scotch_terrier','scottish_deerhound','sealyham_terrier','shetland_sheepdog',
'shih-tzu','siberian_husky','silky_terrier','soft-coated_wheaten_terrier','staffordshire_bullterrier',
'standard_poodle','standard_schnauzer','sussex_spaniel','tibetan_mastiff','tibetan_terrier',
'toy_poodle','toy_terrier','vizsla','walker_hound','weimaraner',
'welsh_springer_spaniel','west_highland_white_terrier','whippet','wire-haired_fox_terrier','yorkshire_terrier'
]

def preprocess_image(image: Image.Image):
    """Resize and normalize image to model input."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def safe_predict(model, image: Image.Image):
    x = preprocess_image(image)
    preds = model.predict(x)
    return preds

model = None

def try_load_full_model_direct():
    """Try loading the saved .keras model with a straightforward custom_objects mapping."""
    print("Attempt: load full .keras model with basic custom_objects mapping...")
    co = {
        "TFSMLayer": hub.KerasLayer,
        "KerasLayer": hub.KerasLayer,
    }
    loaded = tf.keras.models.load_model(FULL_MODEL_PATH, custom_objects=co)
    print("Loaded full model directly.")
    return loaded

def try_load_full_model_with_factory():
    """
    Some saved models expect a TFSMLayer that when reconstructed points to
    the exact filepath. We provide a factory that ignores incoming config
    and returns a KerasLayer bound to our local FEATURE_VECTOR_DIR.
    """
    print("Attempt: load full .keras model with factory for TFSMLayer -> local feature extractor...")
    if not os.path.exists(FEATURE_VECTOR_DIR):
        raise FileNotFoundError(f"Feature vector folder not found at '{FEATURE_VECTOR_DIR}'")

    def tfsm_factory(*args, **kwargs):
        return hub.KerasLayer(FEATURE_VECTOR_DIR, signature="serving_default", trainable=False)
    co = {
        "TFSMLayer": tfsm_factory,
        "KerasLayer": tfsm_factory,
    }
    loaded = tf.keras.models.load_model(FULL_MODEL_PATH, custom_objects=co)
    print("Loaded full model using factory mapping.")
    return loaded

def rebuild_model_and_load_weights():
    """Rebuild architecture in code using local TF Hub feature extractor, then load fallback weights."""
    print("Attempt: rebuild architecture with local Hub KerasLayer and load weights file...")
    if not os.path.exists(FEATURE_VECTOR_DIR):
        raise FileNotFoundError(f"Feature vector folder not found at '{FEATURE_VECTOR_DIR}'")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights file not found at '{WEIGHTS_PATH}'")

    base = hub.KerasLayer(FEATURE_VECTOR_DIR, signature="serving_default", trainable=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    features = base(inputs)
    if isinstance(features, dict):
        if "feature_vector" in features:
            features = features["feature_vector"]
        else:
            features = list(features.values())[0]
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="classifier_head")(features)
    m = tf.keras.Model(inputs=inputs, outputs=outputs)
    m.load_weights(WEIGHTS_PATH)
    print("Rebuilt model and loaded weights successfully.")
    return m

def load_model_with_fallbacks():
    """Try multiple strategies to get a usable model object."""
    last_exc = None
    try:
        return try_load_full_model_direct()
    except Exception as e:
        print("Direct load failed:", str(e))
        last_exc = e
        traceback.print_exc()

    try:
        return try_load_full_model_with_factory()
    except Exception as e:
        print("Factory-load failed:", str(e))
        last_exc = e
        traceback.print_exc()

    try:
        return rebuild_model_and_load_weights()
    except Exception as e:
        print("Rebuild+weights failed:", str(e))
        last_exc = e
        traceback.print_exc()

    raise RuntimeError("All model loading strategies failed. See above for detail.") from last_exc

print("Loading model (this may take a while)...")
model = load_model_with_fallbacks()
print("Model is ready.")

def predict_breed(image: Image.Image):
    if model is None:
        return {"error": "model not loaded"}
    preds = safe_predict(model, image)[0]
    top3 = preds.argsort()[-3:][::-1]
    out = {CLASSES[i]: float(preds[i]) for i in top3}
    return out

title = "Doggo Breed Identifier"
description = ( "Upload a clear photo of a dog. "
               "The model will predict the top 3 breeds and show its confidence.\n\n"
               f"For full source code and details, see my GitHub repository: "
               "https://github.com/JankData/dog-breed-identification-classification" )

iface = gr.Interface(
    fn=predict_breed,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title=title,
    description=description
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
