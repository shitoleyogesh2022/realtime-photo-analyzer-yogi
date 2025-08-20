import streamlit as st
import os
from analyzer import AIPhotoAnalyzer
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from datetime import datetime
import cv2
from PIL import Image
import numpy as np

# App configuration
APP_TITLE = "📸 Real-time Photography Assistant"
APP_DESCRIPTION = "Use your camera to get instant photography feedback!"
UPLOAD_DIR = "uploads"
ANALYSIS_DIR = "analysis"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

class VideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.analyzer = AIPhotoAnalyzer()
        self.last_analysis = None
        self.current_frame = None
        self.frame_skip = 30
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.current_frame = img.copy()
        
        self.frame_count += 1
        if self.frame_count % self.frame_skip == 0:
            try:
                self.last_analysis = self.analyzer.analyze_image(self.current_frame)
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def save_captured_photo(frame, analysis):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(UPLOAD_DIR, f"photo_{timestamp}.jpg")
    analysis_path = os.path.join(ANALYSIS_DIR, f"analysis_{timestamp}.txt")
    
    cv2.imwrite(image_path, frame)
    
    with open(analysis_path, 'w') as f:
        f.write(str(analysis))
    
    return image_path, analysis_path

def display_analysis(analysis):
    if not analysis:
        return
    
    st.write("**Scene Detection:**")
    for scene, prob in analysis['scene_type']:
        st.write(f"- {scene}: {prob:.2%}")
    
    st.write("**Composition Analysis:**")
    comp = analysis['composition_score']
    st.write(f"- Overall Score: {comp['score']:.2%}")
    st.write(f"- Edge Definition: {comp['edge_density']:.2%}")
    st.write(f"- Symmetry: {comp['symmetry']:.2%}")
    st.write(f"- Rule of Thirds: {comp['thirds_alignment']:.2%}")
    
    st.write("**Lighting Analysis:**")
    light = analysis['lighting_quality']
    st.write(f"- Overall Score: {light['score']:.2%}")
    st.write(f"- Brightness: {light['brightness']:.2%}")
    st.write(f"- Contrast: {light['contrast']:.2%}")
    st.write(f"- Color Balance: {light['color_balance']:.2%}")

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    
    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)

    col1, col2 = st.columns(2)
    
    with col1:
        webrtc_ctx = webrtc_streamer(
            key="photography_assistant",
            video_processor_factory=VideoTransformer,
            async_processing=True,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
        )

    with col2:
        if st.button("📸 Capture Photo"):
            if webrtc_ctx.video_processor and webrtc_ctx.video_processor.current_frame is not None:
                try:
                    image_path, analysis_path = save_captured_photo(
                        webrtc_ctx.video_processor.current_frame,
                        webrtc_ctx.video_processor.last_analysis
                    )
                    
                    st.success(f"Photo captured and saved!")
                    
                    captured_image = cv2.imread(image_path)
                    captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
                    st.image(captured_image, caption="Captured Photo", use_column_width=True)
                    
                    with open(image_path, 'rb') as file:
                        st.download_button(
                            label="Download Photo",
                            data=file,
                            file_name=os.path.basename(image_path),
                            mime="image/jpeg"
                        )
                    
                    with open(analysis_path, 'rb') as file:
                        st.download_button(
                            label="Download Analysis Report",
                            data=file,
                            file_name=os.path.basename(analysis_path),
                            mime="text/plain"
                        )
                    
                except Exception as e:
                    st.error(f"Error capturing photo: {str(e)}")
            else:
                st.warning("Camera not ready. Please wait a moment and try again.")

    if webrtc_ctx.video_processor:
        analysis_placeholder = st.empty()
        
        if webrtc_ctx.video_processor.last_analysis:
            with analysis_placeholder.container():
                display_analysis(webrtc_ctx.video_processor.last_analysis)

    with st.expander("About"):
        st.write("""
        This real-time photography assistant helps you:
        - Analyze scene composition
        - Check lighting conditions
        - Monitor technical quality
        - Get instant feedback
        
        Use the camera preview to compose your shot, then click 'Capture Photo' to save it with detailed analysis.
        """)

if __name__ == "__main__":
    main()
