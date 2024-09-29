import streamlit as st
from ocr import extract_text

st.title("OCR Application")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    extracted_text = extract_text("temp_image.jpg")
    
    st.subheader("Extracted Text:")
    st.write(extracted_text)

    keyword = st.text_input("Enter a keyword to search:")
    
    if keyword:
        if keyword.lower() in extracted_text.lower():
            st.success(f"Keyword '{keyword}' found in the text!")
        else:
            st.error(f"Keyword '{keyword}' not found.")
