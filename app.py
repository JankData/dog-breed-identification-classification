import gradio as gr
import tensorflow as tf

model_save_path = "dog_breed_id_model_version2.keras"
loaded_model = tf.keras.models.load_model(model_save_path, compile=False)

with open('doggo_class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

def pred_on_custom_image(image,
                         model: tf.keras.Model = loaded_model,
                         target_size: int = 224,
                         class_names: list = class_names):

    resize = tf.keras.layers.Resizing(height=target_size, width=target_size)
    custom_image_tensor = resize(tf.keras.utils.img_to_array(image))
    custom_image_tensor = tf.expand_dims(custom_image_tensor, axis=0)
    pred_probs = model.predict(custom_image_tensor)[0]
    return {class_names[i]: float(pred_probs[i]) for i in range(len(class_names))}

interface_title = "Doggo Identification"
interface_description = """Upload a clear photo of a dog. 
The model will identify the top 3 breeds and show the confidence of each prediction.
This model was trained with TensorFlow/Keras.
For full source code and details, check out my GitHub:
https://github.com/JankData/dog-breed-identification-classification
"""

interface = gr.Interface(
    fn=pred_on_custom_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    examples=[
        "examples/dog-photo-1.jpeg",
        "examples/dog-photo-2.jpeg",
        "examples/dog-photo-3.jpeg"],
    title=interface_title,
    description=interface_description
)

interface.launch(debug=True)
