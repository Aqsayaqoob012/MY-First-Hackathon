from groq import Groq
import base64
import os
from dotenv import load_dotenv
import json

load_dotenv()

# -------------------------------
# Step 1: Encode Image & Groq Call
# -------------------------------
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_path = "image.webp"
base64_image = encode_image(image_path)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
    {
      "name": "",
      "quantity": 0,
      "price": 0.0
    }
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
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        },
    ],
    temperature=0  # Deterministic output
)

# -------------------------------
# Step 2: Parse JSON Output
# -------------------------------
try:
    data = json.loads(chat_completion.choices[0].message.content.strip())
except json.JSONDecodeError:
    print("Invalid JSON from model")
    data = None

if data is None:
    exit()

# -------------------------------
# Step 3: Expense Categorization
# -------------------------------
CATEGORY_MAP = {
    "APPLE": "Fruits",
    "BANANA": "Fruits",
    "ORANGE": "Fruits",
    "PEAR": "Fruits",
    "GRAPES": "Fruits",
    "STRAWBERRY": "Fruits",
    "BLUEBERRY": "Fruits",
    "KIWI": "Fruits",
    "WATERMELON": "Fruits",
    "LEMON": "Fruits",
    "RASPBERRY": "Fruits",
    "MILK": "Dairy",
    "CHEESE": "Dairy",
    "YOGURT": "Dairy",
    # Add more mappings as needed
}

category_totals = {}
for item in data["items"]:
    name = item["name"].upper()
    quantity = item["quantity"]
    price = item["price"] * quantity  # Total per line
    category = CATEGORY_MAP.get(name, "Other")
    category_totals[category] = category_totals.get(category, 0.0) + price

overall_total = sum(category_totals.values())
category_totals["Overall"] = overall_total

print("\nExpense Categorization:\n")
print(json.dumps(category_totals, indent=2))

# -------------------------------
# Step 4: Spending Analysis
# -------------------------------
# Compute percentage per category
category_percent = {cat: round((total/overall_total)*100,2) 
                    for cat, total in category_totals.items() if cat != "Overall"}

# Overspending threshold
overspending_threshold = 40.0
overspending_categories = [cat for cat, pct in category_percent.items() if pct > overspending_threshold]

analysis = {
    "category_percentage": category_percent,
    "overall_total": overall_total,
    "overspending_categories": overspending_categories
}

print("\nSpending Analysis:\n")
print(json.dumps(analysis, indent=2))



from groq import Groq
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
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": financial_prompt
        }
    ],
    temperature=0.6,
    max_completion_tokens=1024,
    top_p=0.95,
    stream=True
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")



