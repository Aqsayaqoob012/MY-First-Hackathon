import streamlit as st 
import base64
import json
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------
# Set page config & theme
# -------------------------------
st.set_page_config(page_title="Receipt Analyzer & Financial Advisor", layout="wide")

# Custom CSS for black & white theme
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #111111;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #444444;
        color: white;
    }
    .st-expander {
        background-color: #222222;
        color: #ffffff;
        border: 1px solid #555555;
        border-radius: 5px;
        padding: 10px;
    }
    .stImage>img {
        border: 2px solid #ffffff;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🧾 Receipt Analyzer & Financial Advisor")

uploaded_file = st.file_uploader("Upload your receipt image", type=["png","jpg","jpeg","webp"])

if uploaded_file:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    def resize_for_display(img, width=400):
        h, w = img.shape[:2]
        ratio = width / w
        new_h = int(h * ratio)
        return cv2.resize(img, (width, new_h))

    # -------------------------------
    # Step 1: Receipt Image Processing
    # -------------------------------
    with st.expander("Step 1: Receipt Image Processing"):
        st.subheader("Original Image")
        st.image(resize_for_display(img)[:, :, ::-1], use_column_width=False)

        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        st.subheader("Grayscale Conversion")
        st.image(resize_for_display(gray), use_column_width=False)

        # Noise Reduction
        denoised = cv2.GaussianBlur(gray, (5,5), 0)
        st.subheader("Noise Reduction")
        st.image(resize_for_display(denoised), use_column_width=False)

        # Contrast Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        st.subheader("Contrast Enhancement")
        st.image(resize_for_display(contrast), use_column_width=False)

        # Thresholding
        _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        st.subheader("Thresholding")
        st.image(resize_for_display(thresh), use_column_width=False)

        st.success("Image preprocessing applied!")

    # Encode preprocessed image for OCR
    _, buffer = cv2.imencode(".jpg", thresh)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    # -------------------------------
    # Step 2: Data Parsing & Structuring
    # -------------------------------
    with st.expander("Step 2: Data Parsing & Structuring"):
        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a receipt data extraction system.
Return ONLY valid JSON.
Do NOT explain anything.
Do NOT describe the image.
Do NOT add extra text.
If a field is missing, return null.
Format strictly:
{
  "store_name": "",
  "items": [
    {"name": "", "quantity": 0, "price": 0.0}
  ],
  "total": 0.0,
  "cash": 0.0,
  "change": 0.0
}
"""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract receipt data."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0
        )

        raw_output = chat_completion.choices[0].message.content.strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            st.error("Invalid JSON from model")
            st.stop()

        st.json(data)

    # -------------------------------
    # Step 3: Expense Categorization
    # -------------------------------
    with st.expander("Step 3: Expense Categorization"):
        CATEGORY_MAP = {
            "APPLE": "Fruits", "BANANA": "Fruits", "ORANGE": "Fruits",
            "PEAR": "Fruits", "GRAPES": "Fruits", "STRAWBERRY": "Fruits",
            "BLUEBERRY": "Fruits", "KIWI": "Fruits", "WATERMELON": "Fruits",
            "LEMON": "Fruits", "RASPBERRY": "Fruits", "MILK": "Dairy",
            "CHEESE": "Dairy", "YOGURT": "Dairy"
        }

        category_totals = {}
        for item in data["items"]:
            name = item["name"].upper()
            quantity = item["quantity"]
            price = item["price"] * quantity
            category = CATEGORY_MAP.get(name, "Other")
            category_totals[category] = category_totals.get(category, 0.0) + price

        overall_total = sum(category_totals.values())
        category_totals["Overall"] = overall_total
        st.json(category_totals)

    # -------------------------------
    # Step 4: Spending Analysis
    # -------------------------------
    with st.expander("Step 4: Spending Analysis"):
        category_percent = {cat: round((total/overall_total)*100,2)
                            for cat, total in category_totals.items() if cat != "Overall"}
        overspending_categories = [cat for cat, pct in category_percent.items() if pct > 40.0]
        analysis = {
            "category_percentage": category_percent,
            "overall_total": overall_total,
            "overspending_categories": overspending_categories
        }
        st.json(analysis)

    # -------------------------------
    # Step 5: LLM Financial Advice
    # -------------------------------
    with st.expander("Step 5: LLM Financial Advice"):
        financial_prompt = f"""
You are a financial advisor AI.
Here is a receipt's structured data:

{json.dumps(data, indent=2)}

Also, here are the category totals:

{json.dumps(category_totals, indent=2)}

And spending analysis:

{json.dumps(analysis, indent=2)}

Please provide concise, personalized budgeting advice.
Focus on overspending categories.
Return only text.
"""
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": financial_prompt}],
            temperature=0.7,
            max_completion_tokens=1024
        )

        advice = completion.choices[0].message.content.strip()
        st.text(advice)
