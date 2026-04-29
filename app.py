"""
Fake News Detection System - Complete with Combined Analysis
4-Layer Ensemble + Text Analysis + Combined Results + Suggestions
"""
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import re
import os
import numpy as np
from PIL import Image, ImageChops
import io
import requests
import base64
from scipy.fft import fft2
import torch
import torch.nn as nn
from torchvision import transforms, models

# ==================== TEXT MODEL LOAD ====================
@st.cache_resource
def load_text_model():
    model_path = 'models/text_model.pkl'
    if not os.path.exists(model_path):
        return None, None
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data['vectorizer'], data['classifier']
    except:
        return None, None

# ==================== TEXT REASONING WITH SUGGESTIONS ====================
def generate_text_reasoning(text, fake_score):
    text_lower = text.lower()
    reasoning = []
    suggestions = []
    
    sensational = ['breaking', 'urgent', 'shocking', 'viral', 'alert', 'warning', 'miracle', 'unbelievable']
    found = [w for w in sensational if w in text_lower]
    if found:
        reasoning.append(f"⚠️ Sensational language: {', '.join(found[:3])}")
        suggestions.append("✓ Verify claims with official sources before sharing")
    
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.15:
        reasoning.append(f"⚠️ Excessive capitalization ({caps_ratio:.0%} of text)")
        suggestions.append("✓ Legitimate news rarely uses all caps for emphasis")
    
    exclamation_count = text.count('!')
    if exclamation_count > 2:
        reasoning.append(f"⚠️ Multiple exclamations ({exclamation_count} ! marks)")
        suggestions.append("✓ Excessive punctuation often indicates emotional manipulation")
    
    if fake_score > 0.7:
        reasoning.append("🔴 HIGH PROBABILITY OF FAKE NEWS")
        suggestions.append("🚨 Do NOT share this content without verification")
        suggestions.append("✓ Check fact-checking websites (Snopes, FactCheck.org)")
    elif fake_score > 0.5:
        reasoning.append("🟠 SUSPICIOUS - Verify before sharing")
        suggestions.append("⚠️ Cross-reference with multiple trusted news sources")
    elif fake_score > 0.3:
        reasoning.append("🟡 UNCERTAIN - Mixed signals")
        suggestions.append("✓ Verify with trusted sources before sharing")
    else:
        reasoning.append("🟢 LIKELY REAL - Patterns consistent with legitimate news")
        suggestions.append("✅ Content appears legitimate")
    
    return " | ".join(reasoning), suggestions

# ==================== LAYER 1: REALITY DEFENDER API ====================
def layer1_reality_defender(image_file, api_key):
    try:
        image_bytes = image_file.getvalue()
        
        signed_response = requests.post(
            "https://api.prd.realitydefender.xyz/api/files/aws-presigned",
            json={"fileName": "upload.jpg", "fileSize": len(image_bytes)},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=30
        )
        
        if signed_response.status_code != 200:
            return 0.5
        
        signed_data = signed_response.json()
        signed_url = signed_data.get("response", {}).get("signedUrl")
        request_id = signed_data.get("response", {}).get("requestId")
        
        if not signed_url:
            return 0.5
        
        requests.put(signed_url, data=image_bytes, headers={"Content-Type": "image/jpeg"}, timeout=30)
        
        result_response = requests.get(
            f"https://api.prd.realitydefender.xyz/api/media/users/{request_id}",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=30
        )
        
        if result_response.status_code == 200:
            result = result_response.json()
            return result.get('fake_probability', 0.5)
        
        return 0.5
    except Exception as e:
        return 0.5

# ==================== LAYER 2: ELA ANALYSIS ====================
def layer2_ela_analysis(image_file):
    try:
        img = Image.open(image_file).convert('RGB')
        
        quality_levels = [95, 85, 75, 65]
        ela_scores = []
        
        for quality in quality_levels:
            temp_high = io.BytesIO()
            temp_low = io.BytesIO()
            
            img.save(temp_high, format='JPEG', quality=quality)
            img.save(temp_low, format='JPEG', quality=50)
            
            img_high = Image.open(temp_high)
            img_low = Image.open(temp_low)
            
            diff = ImageChops.difference(img_high, img_low)
            diff_array = np.array(diff)
            ela_scores.append(np.mean(diff_array) / 255.0)
        
        ela_score = np.mean(ela_scores)
        fake_score = min(ela_score * 1.8, 0.95)
        
        return fake_score, f"ELA score: {ela_score:.2%}"
    except Exception as e:
        return 0.3, "ELA analysis failed"

# ==================== GENERATE ELA IMAGE ====================
def generate_ela_image(image, quality=90):
    temp_orig = io.BytesIO()
    image.save(temp_orig, format='JPEG', quality=quality)
    temp_orig.seek(0)
    img_recompressed = Image.open(temp_orig)
    
    diff = ImageChops.difference(image, img_recompressed)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_img = Image.eval(diff, lambda px: px * scale)
    
    return ela_img

# ==================== LOCAL EDITS DETECTION ====================
def detect_local_edits_enhanced(image_file):
    try:
        img = Image.open(image_file).convert('RGB')
        img_array = np.array(img)
        
        fake_score = 0.30
        reasons = []
        
        ela_scores = []
        for quality in [95, 85, 75, 65]:
            ela_img = generate_ela_image(img, quality=quality)
            ela_array = np.array(ela_img)
            ela_intensity = np.mean(ela_array) / 255.0
            ela_scores.append(ela_intensity)
        
        avg_ela = np.mean(ela_scores)
        ela_std = np.std(ela_scores)
        
        if ela_std > 0.08:
            fake_score += 0.30
            reasons.append("Inconsistent ELA across qualities")
        elif avg_ela > 0.18:
            fake_score += 0.25
            reasons.append("High ELA intensity detected")
        
        from scipy import ndimage
        gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
        edges = np.abs(ndimage.sobel(gray))
        edge_density = np.mean(edges)
        
        if edge_density > 40:
            fake_score += 0.15
            reasons.append("Unnatural edge patterns")
        
        h, w = gray.shape
        quadrants = [
            gray[:h//2, :w//2],
            gray[:h//2, w//2:],
            gray[h//2:, :w//2],
            gray[h//2:, w//2:]
        ]
        quadrant_vars = [np.var(q) for q in quadrants]
        var_std = np.std(quadrant_vars)
        
        if var_std > 25:
            fake_score += 0.15
            reasons.append("Inconsistent texture across regions")
        
        fake_score = min(fake_score, 0.95)
        
        return fake_score, " | ".join(reasons) if reasons else "No obvious manipulation detected"
        
    except Exception as e:
        return 0.35, "Local edit analysis failed"

# ==================== LAYER 3: NOISE ANALYSIS ====================
def layer3_noise_analysis(image_file):
    try:
        img = Image.open(image_file).convert('RGB')
        img_array = np.array(img)
        
        gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
        
        noise_var = np.var(gray)
        
        if noise_var < 30:
            noise_score = 0.8
            reason_noise = "Very low noise (AI generation likely)"
        elif noise_var < 50:
            noise_score = 0.5
            reason_noise = "Low noise variance (suspicious)"
        elif noise_var > 130:
            noise_score = 0.6
            reason_noise = "High noise (compression artifacts)"
        else:
            noise_score = 0.2
            reason_noise = "Normal noise level"
        
        f_transform = fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        center_region = magnitude[center_h-50:center_h+50, center_w-50:center_w+50]
        total_magnitude = np.mean(magnitude)
        center_magnitude = np.mean(center_region)
        
        if center_magnitude > total_magnitude * 2.0:
            noise_score = max(noise_score, 0.7)
            reason_noise += " | Grid artifacts detected"
        
        return noise_score, reason_noise
    except Exception as e:
        return 0.3, "Noise analysis failed"

# ==================== LAYER 4: METADATA ANALYSIS ====================
def layer4_metadata_analysis(image_file):
    fake_score = 0.2
    reasoning = []
    
    try:
        img = Image.open(image_file)
        width, height = img.size
        aspect = width / height
        
        if aspect > 2.2 or aspect < 0.45:
            fake_score += 0.25
            reasoning.append("Highly unusual aspect ratio")
        elif aspect > 1.8 or aspect < 0.55:
            fake_score += 0.15
            reasoning.append("Unusual aspect ratio")
        
        file_size = len(image_file.getvalue())
        if file_size < 20000:
            fake_score += 0.3
            reasoning.append("Very small file (heavy compression)")
        elif file_size < 50000:
            fake_score += 0.15
            reasoning.append("Small file size")
        
        fake_score = min(fake_score, 0.95)
        
        return fake_score, " | ".join(reasoning) if reasoning else "Normal metadata"
    except Exception as e:
        return 0.2, "Metadata analysis failed"

# ==================== IMAGE REASONING & SUGGESTIONS ====================
def generate_image_reasoning_and_suggestions(result, layer_scores):
    reasoning = []
    suggestions = []
    
    fake_score = result['fake_score']
    verdict = result['class']
    
    if verdict == 'FAKE':
        reasoning.append(f"🔴 VERDICT: FAKE IMAGE")
        suggestions.append("🚨 Do NOT share this image without verification")
        suggestions.append("✓ Try reverse image search on Google Images")
    elif verdict == 'SUSPICIOUS':
        reasoning.append(f"🟠 VERDICT: SUSPICIOUS")
        suggestions.append("⚠️ Be cautious - verify before sharing")
    else:
        reasoning.append(f"🟢 VERDICT: REAL IMAGE")
        suggestions.append("✅ Image appears authentic")
    
    if layer_scores.get('Local Edit Detection', 0) > 0.50:
        reasoning.append("🔍 Local editing detected")
    
    if layer_scores.get('AI/Noise Detection', 0) > 0.75:
        reasoning.append("🔍 AI generation artifacts detected")
    
    return " | ".join(reasoning), suggestions

# ==================== MAIN IMAGE ANALYSIS ====================
def analyze_image_complete(image_file, api_key):
    image_file.seek(0)
    rd_score = layer1_reality_defender(image_file, api_key) if api_key else 0.5
    
    image_file.seek(0)
    local_edit_score, local_edit_reason = detect_local_edits_enhanced(image_file)
    
    image_file.seek(0)
    ela_score, ela_reason = layer2_ela_analysis(image_file)
    
    image_file.seek(0)
    noise_score, noise_reason = layer3_noise_analysis(image_file)
    
    image_file.seek(0)
    meta_score, meta_reason = layer4_metadata_analysis(image_file)
    
    final_score = (rd_score * 0.15) + (local_edit_score * 0.45) + (ela_score * 0.20) + (noise_score * 0.15) + (meta_score * 0.05)
    
    # FINAL BALANCED THRESHOLDS
    if local_edit_score > 0.50 or rd_score > 0.60 or final_score > 0.55:
        verdict = "FAKE"
    elif final_score > 0.40:
        verdict = "SUSPICIOUS"
    else:
        verdict = "REAL"
    
    layer_scores = {
        'Reality Defender (Face)': rd_score,
        'Local Edit Detection': local_edit_score,
        'ELA Analysis': ela_score,
        'AI/Noise Detection': noise_score,
        'Metadata': meta_score
    }
    
    return {
        'fake_score': final_score,
        'class': verdict,
        'confidence': min(0.95, max(0.5, 1 - abs(final_score - 0.5) * 1.5)),
        'layer_scores': layer_scores,
        'ela_reason': ela_reason,
        'noise_reason': noise_reason,
        'meta_reason': meta_reason,
        'local_edit_reason': local_edit_reason
    }

def analyze_image_basic(image_file):
    try:
        img = Image.open(image_file)
        img_array = np.array(img)
        
        fake_score = 0.2
        reasoning = []
        
        h, w = img_array.shape[:2]
        aspect = w / h
        if aspect > 1.8 or aspect < 0.55:
            fake_score += 0.25
            reasoning.append("Unusual aspect ratio")
        
        if len(img_array.shape) == 3:
            avg_std = (np.std(img_array[:,:,0]) + np.std(img_array[:,:,1]) + np.std(img_array[:,:,2])) / 3
            if avg_std < 35:
                fake_score += 0.35
                reasoning.append("Too smooth (AI generation)")
        
        file_size = len(image_file.getvalue())
        if file_size < 30000:
            fake_score += 0.2
            reasoning.append("Very small file")
        
        fake_score = min(fake_score, 0.95)
        
        return {
            'fake_score': fake_score,
            'class': 'FAKE' if fake_score > 0.55 else 'REAL',
            'confidence': 0.7,
            'layer_scores': {'Basic Analysis': fake_score}
        }
    except Exception as e:
        return None

# ==================== GAUGE CHART ====================
def create_gauge_chart(score, title="Fake Score"):
    fig, ax = plt.subplots(figsize=(8, 3))
    
    if score > 0.55:
        color = '#e74c3c'
        status = "High Risk"
    elif score > 0.40:
        color = '#f39c12'
        status = "Medium Risk"
    else:
        color = '#2ecc71'
        status = "Low Risk"
    
    ax.barh([0], [score], color=color, height=0.3)
    ax.barh([0], [1], alpha=0.2, color='gray', height=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(f"{title}: {score*100:.1f}%", fontsize=14, fontweight='bold')
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_yticks([])
    ax.legend(loc='upper right')
    ax.text(score, 0.4, f"{score*100:.0f}%", ha='center', fontsize=12, fontweight='bold')
    ax.text(0.95, -0.3, status, ha='right', fontsize=10, color=color, fontweight='bold')
    
    plt.tight_layout()
    return fig

# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="Fake Content Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ Fake Content Detection System")
st.markdown("*4-Layer Ensemble: Face + Local Edits + AI Artifacts + Metadata*")

vectorizer, classifier = load_text_model()
API_KEY = st.secrets.get("REALITY_DEFENDER_API_KEY", "")

with st.sidebar:
    st.header("📊 System Status")
    
    if vectorizer and classifier:
        st.success("✅ Text Model: Ready")
    else:
        st.error("❌ Text Model: Missing")
    
    if API_KEY:
        st.success("✅ Reality Defender API: Configured")
    else:
        st.warning("⚠️ API: Basic mode (limited detection)")
    
    st.success("✅ ELA: Ready")
    st.success("✅ AI Detection: Ready")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis", "🔗 Combined Analysis"])

# ==================== TAB 1: TEXT ANALYSIS ====================
with tab1:
    st.header("Analyze News Article")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Fake News Example"):
            st.session_state['news_text'] = "URGENT! Breaking news! You won't believe what happened! Share before deleted! Miracle cure that doctors hate!"
    with col2:
        if st.button("📋 Real News Example"):
            st.session_state['news_text'] = "The president announced new economic policies today at the White House according to official sources."
    
    news_text = st.text_area("Enter news article:", height=150, value=st.session_state.get('news_text', ''))
    
    if st.button("Analyze Text", type="primary"):
        if news_text and vectorizer and classifier:
            processed = news_text.lower()
            processed = re.sub(r'[^a-zA-Z\s]', '', processed)
            features = vectorizer.transform([processed])
            proba = classifier.predict_proba(features)[0]
            fake_score = proba[0]
            
            col1, col2 = st.columns(2)
            with col1:
                if fake_score > 0.5:
                    st.error(f"## ⚠️ FAKE NEWS DETECTED")
                else:
                    st.success(f"## ✅ REAL NEWS")
            with col2:
                st.metric("Fake Score", f"{fake_score*100:.1f}%")
            
            st.pyplot(create_gauge_chart(fake_score, "Fake News Probability"))
            plt.close()
            
            reasoning, suggestions = generate_text_reasoning(news_text, fake_score)
            st.info(f"🔍 {reasoning}")
            
            st.subheader("💡 Recommendations")
            for s in suggestions:
                st.write(s)

# ==================== TAB 2: IMAGE ANALYSIS ====================
with tab2:
    st.header("Analyze Image")
    st.caption("Detects: Face swap | Deepfake | Clothes change | AI generated | Photoshop")
    
    uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", width=300)
        
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing with 4-layer ensemble..."):
                if API_KEY:
                    result = analyze_image_complete(uploaded_image, API_KEY)
                else:
                    result = analyze_image_basic(uploaded_image)
                
                if result:
                    if result['class'] == 'FAKE':
                        st.error(f"## ⚠️ FAKE IMAGE DETECTED")
                    elif result['class'] == 'SUSPICIOUS':
                        st.warning(f"## ⚠️ SUSPICIOUS IMAGE")
                    else:
                        st.success(f"## ✅ REAL IMAGE")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Fake Score", f"{result['fake_score']*100:.1f}%")
                    with col2:
                        st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    
                    st.pyplot(create_gauge_chart(result['fake_score'], "Overall Fake Score"))
                    plt.close()
                    
                    with st.expander("📊 Layer-wise Analysis"):
                        for layer, score in result['layer_scores'].items():
                            st.progress(score, text=f"{layer}: {score*100:.1f}%")
                    
                    img_reasoning, img_suggestions = generate_image_reasoning_and_suggestions(result, result.get('layer_scores', {}))
                    st.info(f"🔍 {img_reasoning}")
                    
                    if 'local_edit_reason' in result:
                        st.caption(f"📝 Local Edit Analysis: {result['local_edit_reason']}")
                    
                    st.subheader("💡 Recommendations")
                    for s in img_suggestions:
                        st.write(s)

# ==================== TAB 3: COMBINED ANALYSIS (UPDATED) ====================
with tab3:
    st.header("🔗 Combined Text + Image Analysis")
    st.caption("Most accurate - analyzes both text and image together")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 News Text")
        combined_text = st.text_area(
            "Enter or paste the news article text:",
            height=200,
            key="combined_text",
            placeholder="Example: The government announced new economic policies today..."
        )
        
        # Quick test examples for combined tab
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📋 Fake News Example", use_container_width=True):
                st.session_state['combined_text'] = "URGENT! Breaking news! You won't believe what happened! This is the biggest secret they don't want you to know! Share before deleted!"
                st.rerun()
        with col_b:
            if st.button("📋 Real News Example", use_container_width=True):
                st.session_state['combined_text'] = "The government announced new economic policies today aimed at helping small businesses. The plan includes tax incentives and infrastructure funding according to official sources."
                st.rerun()
        
        # Load example if selected
        if 'combined_text' in st.session_state:
            combined_text = st.session_state['combined_text']
    
    with col2:
        st.subheader("🖼️ Associated Image")
        combined_image = st.file_uploader(
            "Upload the image associated with this news:",
            type=['jpg', 'jpeg', 'png', 'webp'],
            key="combined_image"
        )
        if combined_image:
            st.image(combined_image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Analyze Both (Text + Image)", type="primary", use_container_width=True):
        if combined_text and combined_image:
            if vectorizer and classifier and API_KEY:
                with st.spinner("Analyzing both text and image together..."):
                    
                    # ========== TEXT ANALYSIS ==========
                    processed = combined_text.lower()
                    processed = re.sub(r'[^a-zA-Z\s]', '', processed)
                    features = vectorizer.transform([processed])
                    proba = classifier.predict_proba(features)[0]
                    
                    text_fake_score = proba[0]
                    text_real_score = proba[1]
                    text_confidence = max(proba)
                    
                    # Text Reasoning
                    text_reasons = []
                    text_suggestions = []
                    text_lower = combined_text.lower()
                    
                    # Check for fake indicators
                    sensational_words = ['breaking', 'urgent', 'shocking', 'viral', 'alert', 'warning', 'miracle', 'unbelievable']
                    found_sensational = [w for w in sensational_words if w in text_lower]
                    if found_sensational:
                        text_reasons.append(f"Sensational words: {', '.join(found_sensational[:3])}")
                        text_suggestions.append("✓ Verify claims with official sources")
                    
                    caps_ratio = sum(1 for c in combined_text if c.isupper()) / max(len(combined_text), 1)
                    if caps_ratio > 0.15:
                        text_reasons.append(f"Excessive capitalization ({caps_ratio:.0%} of text)")
                        text_suggestions.append("✓ Legitimate news rarely uses all caps")
                    
                    exclamation_count = combined_text.count('!')
                    if exclamation_count > 2:
                        text_reasons.append(f"Multiple exclamation marks ({exclamation_count} !)")
                        text_suggestions.append("✓ Excessive punctuation indicates manipulation")
                    
                    urgent_words = ['urgent', 'immediately', 'asap', 'now', 'breaking']
                    if any(w in text_lower for w in urgent_words):
                        text_reasons.append("Urgency language detected")
                        text_suggestions.append("✓ Fake news creates false urgency")
                    
                    source_words = ['according to', 'reuters', 'ap', 'bbc', 'cnn', 'official']
                    found_sources = [s for s in source_words if s in text_lower]
                    if not found_sources and len(combined_text.split()) > 50:
                        text_reasons.append("No credible sources cited")
                        text_suggestions.append("✓ Check if news cites verifiable sources")
                    
                    formal_words = ['announced', 'statement', 'official', 'government', 'president', 'minister', 'report', 'study']
                    found_formal = [w for w in formal_words if w in text_lower]
                    if len(found_formal) >= 2:
                        text_reasons.append(f"Formal language detected")
                        text_suggestions.append("✅ Real news uses formal, balanced language")
                    
                    if found_sources:
                        text_reasons.append(f"Source attribution found")
                        text_suggestions.append("✅ Credible sources indicate legitimate reporting")
                    
                    # Text Verdict
                    if text_fake_score > 0.55:
                        text_verdict = "FAKE"
                        text_summary = f"⚠️ TEXT: FAKE ({text_fake_score*100:.1f}% fake)"
                    elif text_fake_score > 0.40:
                        text_verdict = "SUSPICIOUS"
                        text_summary = f"⚠️ TEXT: SUSPICIOUS ({text_fake_score*100:.1f}% fake)"
                    else:
                        text_verdict = "REAL"
                        text_summary = f"✅ TEXT: REAL ({text_real_score*100:.1f}% real)"
                    
                    # ========== IMAGE ANALYSIS ==========
                    combined_image.seek(0)
                    rd_score = layer1_reality_defender(combined_image, API_KEY)
                    
                    combined_image.seek(0)
                    local_edit_score, local_edit_reason = detect_local_edits_enhanced(combined_image)
                    
                    combined_image.seek(0)
                    ela_score, ela_reason = layer2_ela_analysis(combined_image)
                    
                    combined_image.seek(0)
                    noise_score, noise_reason = layer3_noise_analysis(combined_image)
                    
                    combined_image.seek(0)
                    meta_score, meta_reason = layer4_metadata_analysis(combined_image)
                    
                    image_fake_score = (rd_score * 0.20) + (local_edit_score * 0.35) + (ela_score * 0.25) + (noise_score * 0.20)
                    
                    # Image Reasoning
                    image_reasons = []
                    image_suggestions = []
                    
                    if rd_score > 0.55:
                        image_reasons.append(f"Face/Deepfake manipulation detected ({rd_score*100:.1f}%)")
                        image_suggestions.append("✓ The face appears manipulated")
                    
                    if local_edit_score > 0.50:
                        image_reasons.append(f"Local editing detected - possible clothes change")
                        image_suggestions.append("✓ Image shows signs of digital manipulation")
                    
                    if ela_score > 0.45:
                        image_reasons.append(f"Compression artifacts detected")
                        image_suggestions.append("✓ Inconsistent compression suggests editing")
                    
                    if noise_score > 0.55:
                        image_reasons.append(f"AI generation artifacts detected")
                        image_suggestions.append("✓ Image may be AI-generated")
                    
                    if not image_reasons:
                        image_reasons.append("No manipulation detected")
                        image_suggestions.append("✅ Image appears authentic")
                    
                    # Image Verdict
                    if image_fake_score > 0.55:
                        image_verdict = "FAKE"
                        image_summary = f"⚠️ IMAGE: FAKE ({image_fake_score*100:.1f}% fake)"
                    elif image_fake_score > 0.40:
                        image_verdict = "SUSPICIOUS"
                        image_summary = f"⚠️ IMAGE: SUSPICIOUS ({image_fake_score*100:.1f}% fake)"
                    else:
                        image_verdict = "REAL"
                        image_summary = f"✅ IMAGE: REAL ({(1-image_fake_score)*100:.1f}% real)"
                    
                    # ========== COMBINED ANALYSIS (4 SCENARIOS) ==========
                    combined_score = (text_fake_score * 0.5) + (image_fake_score * 0.5)
                    
                    # Display Results
                    st.subheader("📊 Combined Analysis Results")
                    
                    # Show individual scores
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Text Fake Score", f"{text_fake_score*100:.1f}%")
                    with col2:
                        st.metric("Image Fake Score", f"{image_fake_score*100:.1f}%")
                    with col3:
                        st.metric("Combined Score", f"{combined_score*100:.1f}%")
                    
                    st.progress(combined_score)
                    
                    # ========== 4 SCENARIOS HANDLING ==========
                    
                    # SCENARIO 1: Both FAKE
                    if text_verdict == "FAKE" and image_verdict == "FAKE":
                        st.error("## 🔴 COMBINED VERDICT: FAKE")
                        st.markdown("**Both the news text and image are FAKE**")
                        st.info(f"📝 {text_summary}")
                        st.info(f"🖼️ {image_summary}")
                        
                        with st.expander("🔍 Detailed Analysis"):
                            st.markdown("**Why this is FAKE:**")
                            for r in text_reasons:
                                st.write(f"📝 {r}")
                            for r in image_reasons:
                                st.write(f"🖼️ {r}")
                        
                        st.subheader("💡 Recommendations")
                        st.warning("🚨 Do NOT share this content anywhere")
                        st.write("✓ Verify through official news sources")
                        st.write("✓ Report as misinformation if seen on social media")
                        st.write("✓ Check fact-checking websites (Snopes, FactCheck.org)")
                    
                    # SCENARIO 2: Both REAL
                    elif text_verdict == "REAL" and image_verdict == "REAL":
                        st.success("## 🟢 COMBINED VERDICT: REAL")
                        st.markdown("**Both the news text and image appear REAL**")
                        st.info(f"📝 {text_summary}")
                        st.info(f"🖼️ {image_summary}")
                        
                        with st.expander("🔍 Detailed Analysis"):
                            st.markdown("**Why this is REAL:**")
                            for r in text_reasons:
                                st.write(f"📝 {r}")
                            for r in image_reasons:
                                st.write(f"🖼️ {r}")
                        
                        st.subheader("💡 Recommendations")
                        st.success("✅ Content appears legitimate")
                        st.write("✓ Still verify critical claims with official sources")
                        st.write("✓ Cross-check with other trusted news outlets")
                    
                    # SCENARIO 3: Text FAKE, Image REAL (Mismatch)
                    elif text_verdict == "FAKE" and image_verdict == "REAL":
                        st.warning("## 🟠 COMBINED VERDICT: SUSPICIOUS - MISMATCH")
                        st.markdown("**The TEXT is FAKE but the IMAGE is REAL**")
                        st.info(f"📝 {text_summary}")
                        st.info(f"🖼️ {image_summary}")
                        
                        with st.expander("🔍 Detailed Analysis"):
                            st.markdown("**Why this is MISMATCH:**")
                            st.markdown("📝 **Text Analysis (FAKE):**")
                            for r in text_reasons:
                                st.write(f"   • {r}")
                            st.markdown("🖼️ **Image Analysis (REAL):**")
                            for r in image_reasons:
                                st.write(f"   • {r}")
                            st.markdown("---")
                            st.markdown("**Conclusion:** The image may be authentic but the text is fabricated. The image might be unrelated to this fake news.")
                        
                        st.subheader("💡 Recommendations")
                        st.warning("⚠️ The TEXT is FAKE - do NOT believe the news")
                        st.write("✓ The IMAGE appears real, but verify its original source")
                        st.write("✓ Try reverse image search on Google Images")
                        st.write("✓ Check if the image is from a different event")
                    
                    # SCENARIO 4: Text REAL, Image FAKE (Mismatch)
                    elif text_verdict == "REAL" and image_verdict == "FAKE":
                        st.warning("## 🟠 COMBINED VERDICT: SUSPICIOUS - MISMATCH")
                        st.markdown("**The TEXT is REAL but the IMAGE is FAKE**")
                        st.info(f"📝 {text_summary}")
                        st.info(f"🖼️ {image_summary}")
                        
                        with st.expander("🔍 Detailed Analysis"):
                            st.markdown("**Why this is MISMATCH:**")
                            st.markdown("📝 **Text Analysis (REAL):**")
                            for r in text_reasons:
                                st.write(f"   • {r}")
                            st.markdown("🖼️ **Image Analysis (FAKE):**")
                            for r in image_reasons:
                                st.write(f"   • {r}")
                            st.markdown("---")
                            st.markdown("**Conclusion:** The news text appears authentic but the image is manipulated or AI-generated.")
                        
                        st.subheader("💡 Recommendations")
                        st.warning("⚠️ The IMAGE is FAKE - do NOT trust the visual")
                        st.write("✓ The TEXT appears real, but verify the image source")
                        st.write("✓ The image may be AI-generated or photoshopped")
                        st.write("✓ Look for the same image from official sources")
                    
                    # Default case (SUSPICIOUS threshold)
                    else:
                        st.warning("## 🟠 COMBINED VERDICT: SUSPICIOUS")
                        st.markdown("**Content shows mixed or uncertain signals**")
                        st.info(f"📝 {text_summary}")
                        st.info(f"🖼️ {image_summary}")
                        
                        with st.expander("🔍 Detailed Analysis"):
                            if text_reasons:
                                st.markdown("**Text Analysis:**")
                                for r in text_reasons:
                                    st.write(f"📝 {r}")
                            if image_reasons:
                                st.markdown("**Image Analysis:**")
                                for r in image_reasons:
                                    st.write(f"🖼️ {r}")
                        
                        st.subheader("💡 Recommendations")
                        st.warning("⚠️ Be cautious - verify before sharing")
                        st.write("✓ Cross-reference with trusted news sources")
                        st.write("✓ Check fact-checking websites")
                
            elif not vectorizer or not classifier:
                st.error("Text model not loaded properly.")
            elif not API_KEY:
                st.error("API key not configured. Please add REALITY_DEFENDER_API_KEY in secrets.")
            else:
                st.error("Models not loaded properly.")
        
        elif not combined_text:
            st.warning("Please enter some text to analyze")
        elif not combined_image:
            st.warning("Please upload an image to analyze")

st.markdown("---")
st.markdown("🛡️ **Fake Content Detection System** | 4-Layer Ensemble | Combined Analysis | AI-Powered")
