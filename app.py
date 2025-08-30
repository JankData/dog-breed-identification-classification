# app.py (copy-paste this whole file)
import os
import traceback
import numpy as np
from PIL import Image
import gradio as gr
import h5py

import tensorflow as tf
import tensorflow_hub as hub

# -----------------------
# CONFIG - adjust paths if needed
# -----------------------
FULL_MODEL_PATH = "models/194834-27082025-full-resnetv2-Adam.keras"
FEATURE_VECTOR_DIR = "models/resnet-v2-tensorflow2-50-feature-vector-v2"
WEIGHTS_PATH = "models/resnetv2_classifier.weights.h5"
IMG_SIZE = 224

# Paste your full 120-class list here in the exact same order you trained with:
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

# -----------------------
# Utilities
# -----------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def safe_predict(m, image: Image.Image):
    x = preprocess_image(image)
    preds = m.predict(x)
    return preds

def inspect_weights_for_dense_name(weights_path):
    """
    Inspect the HDF5 weights file to find likely final Dense layer name.
    Returns a string like 'dense' or 'classifier_head' or None.
    """
    if not os.path.exists(weights_path):
        return None
    try:
        with h5py.File(weights_path, "r") as f:
            # top-level groups often correspond to layer names for Keras HDF5 weights
            keys = list(f.keys())
            # prefer entries that contain 'dense' or 'classifier' or 'head' in name
            candidates = [k for k in keys if any(s in k.lower() for s in ("dense", "classifier", "head"))]
            if candidates:
                # heuristics: pick the last candidate (often final dense)
                return candidates[-1]
            # fallback: pick last group that contains kernel/bias
            for k in reversed(keys):
                subgroup = f[k]
                if any(x in subgroup.keys() for x in ("kernel:0", "kernel")) or any("kernel" in name for name in f[k].keys()):
                    return k
            # last resort: return last top-level key
            if keys:
                return keys[-1]
    except Exception:
        traceback.print_exc()
    return None

# -----------------------
# Loading strategies
# -----------------------
def try_load_full_model():
    """Try to load the full .keras file (may fail due to Keras version differences)."""
    print("Attempt: load full .keras model (fast path)...")
    try:
        # try without compiling (sometimes helps with mismatched optimizer config)
        loaded = tf.keras.models.load_model(FULL_MODEL_PATH, compile=False, custom_objects={})
        print("✅ Full .keras model loaded.")
        return loaded
    except Exception as e:
        print("❌ Full .keras load failed:", str(e))
        traceback.print_exc()
        return None

def rebuild_with_local_hub_and_load_weights():
    """
    Rebuild model: use local TF-Hub SavedModel (feature extractor), 
    then load classifier weights by-name (skip mismatches).
    """
    print("Attempt: rebuild architecture using local TF-Hub feature extractor...")
    if not os.path.exists(FEATURE_VECTOR_DIR):
        raise FileNotFoundError(f"Feature vector folder not found at '{FEATURE_VECTOR_DIR}'")

    # create KerasLayer from local SavedModel, tell it to return outputs as dict
    # signature_outputs_as_dict=True prevents the 'When using a signature...' error.
    base_layer = hub.KerasLayer(FEATURE_VECTOR_DIR, signature="serving_default",
                                signature_outputs_as_dict=True, trainable=False,
                                input_shape=(IMG_SIZE, IMG_SIZE, 3))

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    features = base_layer(inputs)

    # If the KerasLayer returned a dict, pick 'feature_vector' or the first element
    if isinstance(features, dict):
        if "feature_vector" in features:
            features = features["feature_vector"]
        else:
            # take the first output tensor
            features = list(features.values())[0]

    # Detect the probable dense layer name from the weights file
    dense_name = inspect_weights_for_dense_name(WEIGHTS_PATH) or "dense"
    print(f"Using Dense layer name = '{dense_name}' for the classifier head (auto-detected).")

    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name=dense_name)(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Attempt to load weights. Try strict load first, then by-name/skipping mismatches.
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights file not found at '{WEIGHTS_PATH}'")

    try:
        model.load_weights(WEIGHTS_PATH)
        print("✅ Full weights load succeeded (strict).")
    except Exception as e_full:
        print("⚠️ Full weights load failed:", str(e_full))
        print("Attempting partial load by_name=True, skip_mismatch=True ...")
        try:
            model.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True)
            print("✅ Partial weights load succeeded (by_name, skip_mismatch).")
        except Exception as e_partial:
            print("❌ Partial weights load failed:", str(e_partial))
            traceback.print_exc()
            raise

    return model

# Master loader that tries approaches in sequence
def load_model_with_fallbacks():
    # 1) try full model .keras
    m = try_load_full_model()
    if m is not None:
        return m

    # 2) try rebuild from local hub + weights
    print("Falling back to rebuild+weights strategy...")
    m = rebuild_with_local_hub_and_load_weights()
    return m

# -----------------------
# Startup
# -----------------------
print("Loading model (this may take a while)...")
model = load_model_with_fallbacks()
print("Model loaded and ready.")

# -----------------------
# Prediction and Gradio UI
# -----------------------
def predict_breed(image: Image.Image):
    if model is None:
        return {"error": "model not loaded"}
    preds = safe_predict(model, image)[0]
    top3 = preds.argsort()[-3:][::-1]
    out = {CLASSES[i]: float(preds[i]) for i in top3}
    return out

iface = gr.Interface(
    fn=predict_breed,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Doggo Breed Identifier",
    description="Upload a clear photo of a dog. Model returns top-3 predicted breeds."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
