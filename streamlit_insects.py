import io
import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import v2

def load_image():
    file = st.file_uploader(label = 'Download image')
    #left_co, cent_co, last_co = st.columns(3)
    if file is not None:
        image_data = file.getvalue()
        #with cent_co:
        st.image(image_data, width=350)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def load_model():
    model = torch.load('googlenet_model.pt')[0]
    model.eval()
    return model

def preprocessing(img):
    transform = v2.Compose(
        [v2.ToImage(),
         v2.ToDtype(torch.float32, scale=True),
         v2.Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225))]
    )
    img = transform(img)
    return img

def result_of_model(index_label):
    labels  = torch.load('googlenet_model.pt')[1]
    index_label = int(index_label)
    st.write(list(labels.keys())[index_label])

model = load_model()
st.title('Butterflies and moths classification:butterfly:')
img = load_image()
result = st.button('Recognize the image')
if result:
    x = preprocessing(img)
    index = model(x.unsqueeze(0)).argmax()
    st.write('**Recognition results**')
    result_of_model(index)

