import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

def classify_image(test_image):
    print(test_image.shape)
    print(test_image)
    test_image = cv2.resize(test_image, (224, 224))
    test_image=test_image/255
    test_image=test_image.reshape((224,224,3))
    input_arr = np.array([test_image])  # convert single image to batch
    model = tf.keras.models.load_model('mobilenet_3.h5')
    df = pd.read_csv('modelnames.csv')
    # return df.loc[pred,'Name']
    predictions = model.predict(input_arr)
    print(predictions[0])
    print(np.max(predictions[0]))
    print(df.loc[np.argmax(predictions[0]),'Name'])
    print(np.argmax(predictions))
    return df.loc[np.argmax(predictions[0]),'Name']

gr.Interface(fn=classify_image,
             inputs=gr.Image(label='Upload a photo'),
             outputs=gr.Label(label='Predicted Car'),
             examples=['car1.jpg','car2.jpg','car3.jpg','car4.jpg'],
             title='Car Make Detection',
             theme='dark'
             ).launch(share=True)

