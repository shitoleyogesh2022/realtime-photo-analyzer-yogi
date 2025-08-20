import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
from analyzer import AIPhotoAnalyzer
from config import *

def init_session_state():
    """Initialize session state variables"""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = AIPhotoAnalyzer()
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None

def display_analysis(analysis):
    """Display analysis results in a user-friendly format"""
    if not analysis:
        return

    # Overall Score
    score = analysis['overall_score']
    st.markdown(f"### Overall Score: {score:.0%}")
    st.progress(score)

    # Detailed Analysis in expandable sections
    with st.expander("📊 Detailed Analysis"):
        # Composition
        st.markdown("#### Composition")
        comp = analysis['composition']
        st.markdown(f"- Rule of Thirds: {comp['thirds']:.0%}")
        st.markdown(f"- Symmetry: {comp['symmetry']:.0%}")

        # Lighting
        st.markdown("#### Lighting")
        light = analysis['lighting']
        st.markdown(f"- Brightness: {light['brightness']:.0%}")
        st.markdown(f"- Contrast: {light['contrast']:.0%}")

        # Technical
        st.markdown("#### Technical Quality")
        tech = analysis['technical']
        st.markdown(f"- Sharpness: {tech['sharpness']:.0%}")
        st.markdown(f"- Noise Level: {1 - tech['noise']:.0%}")

def generate_suggestions(analysis):
    """Generate improvement suggestions based on analysis"""
    suggestions = []
    
    if analysis['lighting']['brightness'] < 0.3:
        suggestions.append("🔆 Scene is too dark. Try adding more light.")
    elif analysis['lighting']['brightness'] > 0.7:
        suggestions.append("⚠️ Scene might be overexposed. Reduce brightness.")
        
    if analysis['composition']['thirds'] < 0.4:
        suggestions.append("📐 Try applying the rule of thirds for better composition.")
        
    if analysis['technical']['sharpness'] < 0.4:
        suggestions.append("🎯 Image is not sharp. Check focus or reduce movement.")
    
    return suggestions

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📸",
        layout="wide" if not any(agent in st.user_agent.lower() 
                               for agent in ['android', 'iphone', 'ipad', 'mobile']) 
        else "centered"
    )

    init_session_state()

    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload Photo", "Take Photo"],
        horizontal=True
    )

    if input_method == "Upload Photo":
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.session_state.current_image = image
            st.image(image, use_column_width=True)
            
            if st.button("Analyze Photo"):
                with st.spinner("Analyzing..."):
                    st.session_state.current_analysis = (
                        st.session_state.analyzer.analyze_image(image)
                    )

    else:  # Take Photo
        picture = st.camera_input("Take a picture")
        
        if picture:
            image = Image.open(picture)
            st.session_state.current_image = image
            
            with st.spinner("Analyzing..."):
                st.session_state.current_analysis = (
                    st.session_state.analyzer.analyze_image(image)
                )

    # Display analysis if available
    if st.session_state.current_analysis:
        display_analysis(st.session_state.current_analysis)
        
        suggestions = generate_suggestions(st.session_state.current_analysis)
        if suggestions:
            st.markdown("### 💡 Suggestions for Improvement")
            for suggestion in suggestions:
                st.info(suggestion)

        # Save results
        if st.button("Save Results"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save image
            image_path = os.path.join(UPLOAD_DIR, f"photo_{timestamp}.jpg")
            st.session_state.current_image.save(image_path)
            
            # Save analysis
            analysis_path = os.path.join(ANALYSIS_DIR, f"analysis_{timestamp}.txt")
            with open(analysis_path, 'w') as f:
                f.write(str(st.session_state.current_analysis))
            
            st.success("Results saved successfully!")

    # Help section
    with st.expander("ℹ️ Help"):
        st.markdown("""
        ### How to use this app:
        1. Choose to either upload a photo or take a new one
        2. Wait for the automatic analysis
        3. Review the scores and suggestions
        4. Save the results if desired
        
        ### What we analyze:
        - Composition (rule of thirds, symmetry)
        - Lighting (brightness, contrast)
        - Technical quality (sharpness, noise)
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
