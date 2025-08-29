import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

WEIGHTS_PATH='models/resnetv2_classifier.weights.h5'

NUM_CLASSES=120

CLASSES=[
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

def build_model():
    model = tf.keras.Sequential([
        hub.KerasLayer(
            "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5",
            trainable=False
        ),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    model.load_weights(WEIGHTS_PATH)
    return model

model = build_model()

def predict_breed(img: Image.Image):
    img = img.resize((224, 224))
    x = np.array(img)/255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    results = {CLASSES[i]: float(preds[i])*100 for i in top3_idx}
    return results

title = "Dog Breed Identifier 🐶"
description = (
    "Upload a clear photo of a dog. "
    "The model will predict the top 3 breeds and show its confidence.\n\n"
    f"For full source code and details, see my GitHub repository: "
    "https://github.com/JankData/dog-breed-identification-classification"
)

iface = gr.Interface(
    fn=predict_breed,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title=title,
    description=description
)

if __name__ == "__main__":
    iface.launch()
