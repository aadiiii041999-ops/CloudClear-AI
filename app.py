import streamlit as st
import cv2
import numpy as np
import base64

def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64("satellite.jpg.png")

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    
}}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.15);
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🛰️☁️ CloudClear AI")


st.markdown("""
<div style="
display:inline-block;
padding:10px 15px;
background:rgba(0,100,255,0.3);
border-radius:10px;
color:white;
margin-bottom:20px;">
📷 Upload a satellite image to analyze cloud coverage
</div>
""", unsafe_allow_html=True)



col1, col2 = st.columns([1.2,2])

with col1:
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"])
if uploaded:

    file_bytes = np.asarray(
        bytearray(uploaded.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        200,
        255,
        cv2.THRESH_BINARY
    )

    result = cv2.inpaint(
        image,
        mask,
        7,
        cv2.INPAINT_TELEA
    )

    cloud_percent = (
        np.sum(mask > 0) /
        mask.size
    ) * 100

    st.metric(
        "Cloud Coverage %",
        f"{cloud_percent:.2f}%"
    )

    if cloud_percent > 30:
        st.warning("High Cloud Coverage")
    else:
        st.success("Low Cloud Coverage")

    st.subheader("Original Image")
    st.image(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    )

    st.subheader("Cloud Mask")
    st.image(mask)

    st.subheader("Cloud Removed")
    st.image(
        cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )
    )

    _, buffer = cv2.imencode(".jpg", result)

    st.download_button(
        label="Download Processed Image",
        data=buffer.tobytes(),
        file_name="cloud_removed.jpg",
        mime="image/jpeg"
    )