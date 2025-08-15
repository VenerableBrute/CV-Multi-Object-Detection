import streamlit as st
import pandas as pd
from tempfile import NamedTemporaryFile
import os
from tracker import process_video

st.title("🚗 YOLOv8 + DeepSORT Object Tracking Dashboard")
st.write("Upload a video, and get processed tracking with analytics & heatmap.")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        st.info("Processing video... Please wait.")
        video_path, csv_path, heatmap_path = process_video(tmp_path)
        st.success("✅ Processing complete!")

        st.subheader("🎥 Processed Video")
        st.video(video_path)

        st.subheader("🔥 Heatmap")
        st.image(heatmap_path, caption="Object presence heatmap")

        st.subheader("📊 Tracking Data")
        df = pd.read_csv(csv_path)
        st.dataframe(df)

        with open(csv_path, "rb") as f:
            st.download_button("Download CSV", f, file_name="tracking_log.csv")

        with open(video_path, "rb") as f:
            st.download_button("Download Processed Video", f, file_name="output_processed.mp4")

        with open(heatmap_path, "rb") as f:
            st.download_button("Download Heatmap", f, file_name="heatmap.png")

    finally:
        
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
