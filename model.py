import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
import asyncio

def load_image(image_file):
    image = Image.open(image_file)
    return np.array(image)

def detect_text(image):
    reader = easyocr.Reader(['en'])  # Load OCR model for English
    results = reader.readtext(image)
    
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        
        # Draw green box
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        
        # Put green text
        cv2.putText(image, text, (top_left[0], top_left[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    return image, results

def main():
    # Fix "RuntimeError: no running event loop"
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    st.title("OCR Text Detection App")
    st.write("Upload an image to detect text.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Detect Text"):
            with st.spinner("Detecting..."):
                processed_image, results = detect_text(image.copy())
                
                st.image(processed_image, caption="Detected Text", use_container_width=True)
                st.write("### Extracted Text:")
                for _, text, _ in results:
                    st.write(f"- {text}")

if __name__ == "__main__":
    main()
