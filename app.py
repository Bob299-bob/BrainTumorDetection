import streamlit as st
import numpy as np
from keras.utils import load_img, img_to_array
#to resize image due to adjust
from PIL import Image
tumor_image=Image.open('images.jpeg')
Home_image=tumor_image.resize((800,250))
@st.cache_resource
def load_my_model():
    import os
    import gdown
    from tensorflow.keras.models import load_model

    MODEL_PATH = "brain_tumor_model.h5"

    if not os.path.exists(MODEL_PATH):
        file_id = "1uLd8lWvD8_tlVdZYNQQNi06bMtd4PTFh"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)

    return load_model(MODEL_PATH)

model = load_my_model()
st.markdown("<h1 style='color:red;'>Brain_Tumor_Detection</h1>",unsafe_allow_html=True)
uploaded_file=st.file_uploader("upload Brain MRI image",type=['jpg','jpeg','png'])
if(st.button('Predict')):
    if uploaded_file is not None:
        img=load_img(uploaded_file,target_size=(200,200))
        img_array=img_to_array(img)
        new_img=np.expand_dims(img_array,axis=0)
        scale_img=new_img/255.0
        prediction=model.predict(scale_img)
        class_names=['glioma','meningioma','notumor','pituitary','unknown']
        result=class_names[np.argmax(prediction)]
        st.image(uploaded_file,caption='Uploaded image',width=300)
        #Giving threshold to handle random photos 
        confidence=np.max(prediction)*100
        
        if(result=='unknown'):
            st.error('If you have uploaded a wrong image then correct it ,otherwise wait.....')
        elif(confidence<40):
            st.warning('Low Confidence prediction')
        else:
            if(result=='notumor'):
                st.success('No Tumor Detected')
            else:
                st.success(f"Tumor type:- {result}")
        print(confidence)
    else:
        st.error("Kindly Upload Brain MRI")
# Sidebar
st.sidebar.markdown("<h1 style='color:red;'>Brain Tumor Detector</h1>",
                    unsafe_allow_html=True)
st.sidebar.title("Menu")
option = st.sidebar.selectbox(
    "Choose Option",
    ["Home", "About", "Contact"]
)

if option == "Home":
    st.header("Welcome To Brain Tumor Detection System")
    st.write("""         
    Welcome to our Artificial Intelligence powered Brain Tumor Detection System.
    This project uses Deep Learning and Medical Image Analysis to detect brain tumors
    from MRI scan images quickly and accurately.
    """)
    st.image(
        Home_image,
        caption='Brain_Tumor_detection',
        width=600,
    )
elif option == "About":
    st.header("About Us")
    st.write("""
    We are passionate developers and AI enthusiasts working on
    healthcare technology solutions using Deep Learning and Computer Vision.
    The main goal of this project is to use Artificial Intelligence
    for assisting in brain tumor diagnosis through MRI image classification.
    """)
    about_img=Image.open('about.png')
    about_img=about_img.resize((800,280))
    st.image(
        about_img,
        width=600
    )
elif option == "Contact":
    st.header("Contact Page")
    st.markdown("""
    ### 📧 Email
    bobygupta@gmail.com

    ### 💻 GitHub
    https://github.com/Bob299-bob

    ### 💼 LinkedIn
    https://www.linkedin.com/in/boby-gupta-76bb6213a/

    ### 📍 Location
    Lucknow, India
    """)
