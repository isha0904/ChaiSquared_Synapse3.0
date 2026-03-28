import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import time
import re
# Cache reader
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)


reader = load_reader()

st.markdown(
    """
    <style>
    .stApp {
        background-color: #1a4d1a;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
RIC_INFO = {
    "1": {
        "name": "PET / PETE (Polyethylene Terephthalate)",
        "properties": "Clear, strong, lightweight, moisture-resistant",
        "uses": "Water bottles, soda bottles, food containers",
        "safety": "Safe for single use; heat may cause chemical leaching",
        "recycling": "Widely recycled into fibers, carpets, containers",
        "recyclable": "Yes"
    },
    "2": {
        "name": "HDPE (High-Density Polyethylene)",
        "properties": "Strong, stiff, chemical-resistant",
        "uses": "Milk jugs, detergent bottles, toys",
        "safety": "Very safe, low chemical leaching",
        "recycling": "Highly recyclable into pipes, plastic lumber",
        "recyclable": "Yes"
    },
    "3": {
        "name": "PVC (Polyvinyl Chloride)",
        "properties": "Rigid or flexible, weather-resistant",
        "uses": "Pipes, cables, wraps",
        "safety": "Contains harmful chemicals (phthalates)",
        "recycling": "Difficult to recycle",
        "recyclable": "No"
    },
    "4": {
        "name": "LDPE (Low-Density Polyethylene)",
        "properties": "Flexible, soft, lightweight",
        "uses": "Plastic bags, wraps, squeeze bottles",
        "safety": "Safe but not heat-resistant",
        "recycling": "Limited recycling (store drop-off)",
        "recyclable": "No"
    },
    "5": {
        "name": "PP (Polypropylene)",
        "properties": "Strong, heat-resistant",
        "uses": "Yogurt cups, caps, containers",
        "safety": "Safe for reuse and microwaving",
        "recycling": "Increasingly recyclable",
        "recyclable": "Yes"
    },
    "6": {
        "name": "PS (Polystyrene)",
        "properties": "Lightweight, insulating",
        "uses": "Foam cups, packaging",
        "safety": "May release harmful styrene",
        "recycling": "Very difficult to recycle",
        "recyclable": "No"
    },
    "7": {
        "name": "OTHER (Mixed Plastics)",
        "properties": "Mixed or multilayer plastics",
        "uses": "Snack wrappers, bottles, electronics",
        "safety": "May contain BPA or unknown chemicals",
        "recycling": "Almost never recyclable",
        "recyclable": "No"
    }
}


# 🔥 ULTIMATE TRIANGLE DETECTOR
def detect_triangle_regions_enhanced(img):
    """Detect ALL possible RIC triangle shapes"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    regions = []
    
    # Method 1: Contour-based (original)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 200 < area < 10000:  # Reasonable triangle size
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            
            # Triangle or quadrilateral (some RIC have arrows)
            if len(approx) in [3, 4]:
                x, y, w, h = cv2.boundingRect(cnt)
                # Triangle-like aspect ratio
                if 0.5 < w/h < 2.0:
                    regions.append((x, y, w, h, 'contour'))
    
    # Method 2: Hough Circles (for circular RIC variants)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=15, maxRadius=80)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            regions.append((x-r, y-r, r*2, r*2, 'circle'))
    
    # Method 3: MSER (Maximally Stable Extremal Regions) for text regions
    mser = cv2.MSER_create()
    regions_mser, _ = mser.detectRegions(gray)
    for region in regions_mser:
        x, y, w, h = cv2.boundingRect(region)
        if 100 < w*h < 5000 and 0.3 < w/h < 3:
            regions.append((x, y, w, h, 'mser'))
    
    return regions[:10]  # Limit to top 10 candidates

# 🔥 ENHANCED OCR with multiple pipelines
def extract_ric_enhanced(crop):
    """Try every possible OCR method"""
    
    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return None
    
    # 6 different preprocessing methods
    pipelines = [
        # 1. Adaptive threshold (best for uneven lighting)
        lambda x: cv2.adaptiveThreshold(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 
                                       255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2),
        
        # 2. Otsu threshold
        lambda x: cv2.threshold(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 0, 255, 
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        
        # 3. High contrast grayscale
        lambda x: cv2.convertScaleAbs(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 
                                     alpha=3.0, beta=50),
        
        # 4. Morphological operations
        lambda x: cv2.morphologyEx(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 
                                  cv2.MORPH_CLOSE, np.ones((3,3), np.uint8)),
        
        # 5. Bilateral filter + threshold
        lambda x: cv2.threshold(cv2.bilateralFilter(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 9, 75, 75), 127, 255, cv2.THRESH_BINARY)[1],
        
        # 6. Edge-enhanced
        lambda x: cv2.Canny(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 50, 150)
    ]
    
    for i, preprocess in enumerate(pipelines):
        try:
            processed = preprocess(crop)
            
            # Upscale for better OCR
            processed = cv2.resize(processed, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            
            # OCR with digit restriction
            results = reader.readtext(processed, 
                                    detail=1,
                                    allowlist='01234567',
                                    paragraph=False,
                                    width_ths=0.8,
                                    height_ths=0.8)
            
            for (bbox, text, conf) in results:
                if conf > 0.3:  # Minimum confidence
                    digits = ''.join(re.findall(r'\d', text))
                    if digits and digits in RIC_INFO:
                        return digits
                        
        except:
            continue
    
    return None

# 🎯 MAIN DETECTION FUNCTION
def detect_ric_supercharged(image):
    img = np.array(image)
    output = img.copy()
    # 🔥 STEP 0: FULL IMAGE TEXT ANALYSIS (HIGHEST PRIORITY)
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_full = cv2.resize(gray_full, None, fx=2, fy=2)

    results_full = reader.readtext(gray_full)

    full_text = " ".join([text for (_, text, _) in results_full]).lower()

    # 🔥 RULE: "OTHER" → RIC 7 (OVERRIDE EVERYTHING)
    if "other" in full_text:
        cv2.putText(output, "RIC #7 (OTHER detected)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,255), 3)

        return Image.fromarray(output), "7", RIC_INFO["7"]
    # Get all candidate regions
    regions = detect_triangle_regions_enhanced(image)
    
    best_ric = None
    best_conf = 0
    
    # Test every region
    for (x, y, w, h, method) in regions:
        crop = img[y:y+h, x:x+w]
        ric = extract_ric_enhanced(crop)
        
        if ric:
            # Higher confidence for contour-detected regions
            conf = 0.9 if method == 'contour' else 0.7
            if conf > best_conf:
                best_conf = conf
                best_ric = ric
                
                # Draw BEST detection
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(output, f" RIC #{ric}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
    
    # 🔥 FALLBACK: scan full image if nothing found
        if best_ric is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.convertScaleAbs(gray, alpha=2.5, beta=40)
            gray = cv2.resize(gray, None, fx=2, fy=2)

            results = reader.readtext(gray, allowlist='01234567')

            for (bbox, text, conf) in results:
                if conf > 0.4:
                    digits = ''.join(re.findall(r'\d', text))
                    if digits in RIC_INFO:
                        best_ric = digits

                        (tl, tr, br, bl) = bbox
                        tl = tuple(map(int, tl))
                        br = tuple(map(int, br))

                        cv2.rectangle(output, tl, br, (255,0,0), 2)
                        cv2.putText(output, f"RIC #{digits} (fallback)",
                                    (tl[0], tl[1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255,0,0), 2)

                        break

        label = RIC_INFO.get(best_ric, "Unknown")
        return Image.fromarray(output), best_ric, label

st.set_page_config(layout="wide")
st.title("♻️ Smart RIC Detector")
st.write("Detect plastic type and recyclability")

with st.sidebar:
    st.header(" Upload")

    file = st.file_uploader(
        "Upload Image or Video",
        type=["jpg", "png", "jpeg", "webp", "mp4", "avi"]
    )

if file:
    file_type = file.type

    col1, col2 = st.columns(2)

    # =========================
    # 📷 IMAGE INPUT
    # =========================
    if "image" in file_type:
        image = Image.open(file)

        with col1:
            st.subheader(" Input")
            st.image(image, width=300)

        with col2:
            st.subheader(" Result")

            with st.spinner("Detecting RIC..."):
                _, ric_code, ric_label = detect_ric_supercharged(image)

            if ric_code:
                st.markdown(f"## ♻️ RIC #{ric_code}")
                st.markdown(f"### {ric_label}")

                status_emoji = "" if ric_code in ["1","2","5"] else "NO"
                st.markdown(f"**Recyclability:** {status_emoji}")

            else:
                st.error("RIC not detected ")
    # =========================
    # 🎥 VIDEO INPUT (basic)
    # =========================
    elif "video" in file_type:
        with col1:
            st.subheader(" Input Video")
            st.video(file)

        with col2:
            st.subheader(" Result")
            st.info("Video processing: extracting frames...")

            tfile = open("temp.mp4", "wb")
            tfile.write(file.read())

            cap = cv2.VideoCapture("temp.mp4")

            found = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                result_img, ric_code, ric_label = detect_ric_supercharged(img)

                if ric_code:
                    st.success(f"RIC #{ric_code}")
                    st.info(ric_label)
                    st.image(result_img, width="stretch")
                    found = True
                    break

            cap.release()

            if not found:
                st.error("RIC not detected in video")
    # ═══ RESULTS ═══
    st.markdown("---")

    if ric_code:
        st.markdown("##  SUCCESS! RIC Code Detected")
        st.markdown(f"### RIC #{ric_code}")

        # 🔥 If using detailed dictionary
        info = RIC_INFO[ric_code]

        st.markdown(f"**Type:** {info['name']}")
        st.markdown(f"**Properties:** {info['properties']}")
        st.markdown(f"**Common Uses:** {info['uses']}")
        st.markdown(f"**Safety:** {info['safety']}")
        st.markdown(f"**Recycling:** {info['recycling']}")

        # ✅ FINAL recyclable output (YES / NO)
        recyclable = "Yes" if ric_code in ["1", "2", "5"] else "No"
        st.markdown(f"### ♻️ Recyclable: {recyclable}")

        st.balloons()

    else:
        st.markdown("## ❌ No RIC detected")
        st.info("""
    **Tips for better results:**
    - Get CLOSE to the recycling triangle
    - Good lighting, no shadows/glare  
    - Number (1-7) must be visible inside triangle
    - Try different angles if first attempt fails
    """)

# Reference guide
with st.expander("RIC Quick Reference", expanded=False):
    for code, info in RIC_INFO.items():
        st.markdown(f"**#{code}:** {info}")

st.markdown("---")
st.caption(" Powered by EasyOCR + OpenCV | Triangle + Multi-OCR Pipeline")