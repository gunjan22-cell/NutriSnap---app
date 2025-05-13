import streamlit as st
import sqlite3
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
import requests
from PIL import Image
import pandas as pd
import base64

import openai
from openai import OpenAI


client = OpenAI(api_key=st.secrets["openai"]["api_key"])




# Load MobileNetV2 model
model = MobileNetV2(weights="imagenet")

# Set Streamlit page config
st.set_page_config(page_title="NutriSnap", page_icon="🥗", layout="wide")

# Add missing column 'feedback' to the feedback table if it doesn't exist






# Connect to your database
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

# Step 1: Create 'users' table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT
)
""")

# Step 2: Add missing columns if not already there
try:
    cursor.execute("ALTER TABLE users ADD COLUMN goal TEXT")
except sqlite3.OperationalError:
    pass

try:
    cursor.execute("ALTER TABLE users ADD COLUMN region TEXT")
except sqlite3.OperationalError:
    pass

try:
    cursor.execute("ALTER TABLE users ADD COLUMN allergies TEXT")
except sqlite3.OperationalError:
    pass

conn.commit()



import streamlit as st
from PIL import Image

logo = Image.open("C:/Users/gunja/Downloads/download (1).png")
st.sidebar.image(logo, width=120)
st.sidebar.markdown("<h2 style='text-align: center; color: #4f4f4f;'>NutriSnap</h2>", unsafe_allow_html=True)



# Initialize login session
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Sidebar Navigation
selected_option = st.sidebar.radio("Navigation", 
                                  ["Home", "Login", "Register", "Food Detection", "Dietary Management", "Contact Us"])

# Apply custom background CSS based on page
if selected_option == "Login" or selected_option == "Register":
    st.markdown("""
        <style>
            .stApp {
                background-color: #808080;  /* Dark Gray Background */
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            .stApp {
                background: url('https://images.creativemarket.com/0.1.0/ps/3683908/1820/1213/m1/fpnw/wm1/m74so5ylty5cq0la0ib6nxv0j1hbxyehlevasr0ffyzr4rf0shwldskrgw56igaj-.jpg?1512399171&s=fc09780d4ca2f5643af18a79811f5897') no-repeat center center fixed;
                background-size: cover;
            }
        </style>
    """, unsafe_allow_html=True)

# Nutrition box styling
st.markdown("""
    <style>
        .nutrition-box {
            background-color: white;
            color: black;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            font-size: 16px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Page Routing Logic
if selected_option == "Home":
    st.write("Welcome to the Home page!")
elif selected_option == "Login":
    st.write("Login Form Here")
elif selected_option == "Register":
    st.write("Registration Form Here")
elif selected_option == "Food Detection":
    st.write("Food Detection Page")
elif selected_option == "Dietary Management":
    st.write("Dietary Management Page")
elif selected_option == "Contact Us":
    st.write("Contact Us Page")

# Title
st.markdown("<h1 style='text-align: center; color: white;'>NutriSnap: Food Detection And Dietary Management System</h1>", unsafe_allow_html=True)

# Logout button if logged in
if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.success("✅ You have been logged out. Redirecting to Home...")
        st.experimental_rerun()


# Home Page
def home():
    st.title("🍽️ Welcome to NutriSnap")
    st.markdown("""
    **NutriSnap** is an advanced **Food Detection and Dietary Management System** that utilizes **AI-powered food recognition** to detect multiple food items in a dish.

    ### 🔥 Key Features
    ✅ **AI-Powered Food Detection** – Detects multiple food items in a single dish with high accuracy.  
    ✅ **Live Camera & Image Upload** – Supports both real-time food detection and uploaded images.  
    ✅ **Nutritional Analysis** – Displays calories, macronutrients, and ingredient details.  
    ✅ **Personalized Dietary Management** – Recommends meals based on user preferences and dietary restrictions.  
    ✅ **User Authentication & Profiles** – Secure login, registration, and user-specific diet tracking.  
    ✅ **Weekly & Monthly Nutritional Trends** – Helps users track and improve their eating habits. 
    
    **Stay healthy with NutriSnap!** 💚
    """)

# 📦 Dummy get_today_diet function for demonstration
def get_today_diet(goal, region, allergies):
    # In real use, this would be dynamic — pulled from a model or database
    sample_diets = {
        "Weight Loss": {
            "South Indian": {
                "breakfast": "Idli with sambar",
                "lunch": "Brown rice with dal and sabzi",
                "dinner": "Vegetable upma"
            },
            "North Indian": {
                "breakfast": "Poha",
                "lunch": "2 chapatis, dal, salad",
                "dinner": "Vegetable soup and roti"
            }
        },
        "Muscle Gain": {
            "South Indian": {
                "breakfast": "Masala dosa with chutney",
                "lunch": "Chicken biryani and curd",
                "dinner": "Paneer dosa and milkshake"
            },
            "North Indian": {
                "breakfast": "Paratha with paneer",
                "lunch": "Rice, rajma, chicken curry",
                "dinner": "Egg curry with chapatis"
            }
        }
    }

    # Default to avoid KeyError if region/goal not found
    diet = sample_diets.get(goal, {}).get(region, {
        "breakfast": "Oats",
        "lunch": "Mixed veg with roti",
        "dinner": "Soup with toast"
    })

    # Allergy filtering (basic demo logic)
    if allergies and allergies.lower() in diet['lunch'].lower():
        diet['lunch'] += " (⚠️ contains your allergy)"

    return diet


def login():
    st.subheader("🔐 Login")
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        show_password = st.checkbox("Show password", key="show_password")
        if show_password:
            password = st.text_input("Password", type="default", value=password, key="password_visible")

        remember_me = st.checkbox("Remember me", key="remember_me")

        if st.button("Login", key="login_button"):
            cursor.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
            user = cursor.fetchone()
            if user:
                st.success("Login successful!")
                st.session_state.logged_in = True

                # ✅ Fetch user preferences for personalized diet
                cursor.execute("SELECT goal, region, allergies FROM users WHERE email=?", (email,))
                user_info = cursor.fetchone()
                if user_info:
                    goal, region, allergies = user_info

                    # 🔄 Generate personalized diet
                    today_diet = get_today_diet(goal, region, allergies)

                    # ✅ Create diet message
                    diet_message = f"""
🍽️ Hello {email}, here's your personalized diet for today:
- 🥣 Breakfast: {today_diet.get('breakfast', 'N/A')}
- 🍛 Lunch: {today_diet.get('lunch', 'N/A')}
- 🍲 Dinner: {today_diet.get('dinner', 'N/A')}
"""
                    st.toast(diet_message)

            else:
                st.error("Invalid email or password")

        with st.expander("🔒 Forgot Password?"):
            reset_email = st.text_input("Registered Email", key="reset_email")
            new_password = st.text_input("New Password", type="password", key="new_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            reset_btn = st.button("Reset Password", key="reset_btn")

            if reset_btn:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    cursor.execute("SELECT * FROM users WHERE email=?", (reset_email,))
                    user = cursor.fetchone()
                    if user:
                        cursor.execute("UPDATE users SET password=? WHERE email=?", (new_password, reset_email))
                        conn.commit()
                        st.success("✅ Password reset successful!")
                    else:
                        st.error("❌ Email not found.")

# Register Page
def register():
    st.subheader("📝 Register")
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        # New input fields
        first_name = st.text_input("First Name", key="register_fname")
        last_name = st.text_input("Last Name", key="register_lname")
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")

        if st.button("Register"):
            if not first_name or not last_name or not email or not password:
                st.error("Please fill in all the fields.")
            else:
                try:
                    cursor.execute("""
                        INSERT INTO users (first_name, last_name, email, password)
                        VALUES (?, ?, ?, ?)
                    """, (first_name, last_name, email, password))
                    conn.commit()
                    st.success("Registration successful! Please login.")
                except sqlite3.IntegrityError:
                    st.error("Email already registered.")


# Nutritionix API
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the API credentials from environment variables
API_ID = os.getenv("NUTRITIONIX_APP_ID")
API_KEY = os.getenv("NUTRITIONIX_APP_KEY")
API_URL = "https://trackapi.nutritionix.com/v2/natural/nutrients"

def get_nutrition(food_item):
    headers = {
        "x-app-id": API_ID,
        "x-app-key": API_KEY,
        "Content-Type": "application/json"
    }
    data = {"query": food_item}
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json() if response.status_code == 200 else None


# Nutrition data dictionary
nutrition_data = {
    "Ice Cream": {"Calories (kcal)": 273.24, "Protein (g)": 4.62, "Carbs (g)": 31.15, "Fats (g)": 14.52},
    "Pizza": {"Calories (kcal)": 285, "Protein (g)": 12, "Carbs (g)": 36, "Fats (g)": 10},
    "Burger": {"Calories (kcal)": 295, "Protein (g)": 17, "Carbs (g)": 30, "Fats (g)": 14},
    "Chicken": {"Calories (kcal)": 239, "Protein (g)": 27, "Carbs (g)": 0, "Fats (g)": 14},
    "Aloo Paratha": {"Calories (kcal)": 210, "Protein (g)": 5, "Carbs (g)": 30, "Fats (g)": 8},
    "Paneer Tikka": {"Calories (kcal)": 270, "Protein (g)": 19, "Carbs (g)": 9, "Fats (g)": 18},
    "Veg Biryani": {"Calories (kcal)": 250, "Protein (g)": 6, "Carbs (g)": 42, "Fats (g)": 7},
    "Palak Paneer": {"Calories (kcal)": 300, "Protein (g)": 12, "Carbs (g)": 14, "Fats (g)": 22},
    "Rajma Chawal": {"Calories (kcal)": 320, "Protein (g)": 12, "Carbs (g)": 50, "Fats (g)": 8},
    "Dosa": {"Calories (kcal)": 168, "Protein (g)": 4, "Carbs (g)": 30, "Fats (g)": 3.7},
    "Idli": {"Calories (kcal)": 39, "Protein (g)": 1.6, "Carbs (g)": 7.4, "Fats (g)": 0.2},
    "Samosa": {"Calories (kcal)": 308, "Protein (g)": 6, "Carbs (g)": 34, "Fats (g)": 17},
    "Chana Chaat": {"Calories (kcal)": 180, "Protein (g)": 9, "Carbs (g)": 28, "Fats (g)": 3},
    "Chole Bhature": {"Calories (kcal)": 450, "Protein (g)": 13, "Carbs (g)": 45, "Fats (g)": 25},
    "Poha": {"Calories (kcal)": 180, "Protein (g)": 4, "Carbs (g)": 30, "Fats (g)": 6},
    "Upma": {"Calories (kcal)": 200, "Protein (g)": 5, "Carbs (g)": 32, "Fats (g)": 7}
}
nutrition_data.update({
    "cannoli": {
        "calories": 230,
        "protein": "5g",
        "fat": "13g",
        "carbohydrates": "25g",
        "fiber": "1g",
        "sugar": "14g"
    },
    "falafel": {
        "calories": 330,
        "protein": "13g",
        "fat": "17g",
        "carbohydrates": "30g",
        "fiber": "8g",
        "sugar": "2g"
    },
    "edamame": {
        "calories": 120,
        "protein": "11g",
        "fat": "5g",
        "carbohydrates": "9g",
        "fiber": "4g",
        "sugar": "2g"
    },
    "french toast": {
        "calories": 280,
        "protein": "10g",
        "fat": "12g",
        "carbohydrates": "30g",
        "fiber": "2g",
        "sugar": "10g"
    },
    "ice cream": {
        "calories": 210,
        "protein": "4g",
        "fat": "11g",
        "carbohydrates": "24g",
        "fiber": "0g",
        "sugar": "21g"
    },
    "ramen": {
        "calories": 430,
        "protein": "10g",
        "fat": "15g",
        "carbohydrates": "60g",
        "fiber": "2g",
        "sugar": "1g"
    },
    "sushi": {
        "calories": 200,
        "protein": "9g",
        "fat": "4g",
        "carbohydrates": "35g",
        "fiber": "2g",
        "sugar": "5g"
    },
    "tiramisu": {
        "calories": 300,
        "protein": "5g",
        "fat": "18g",
        "carbohydrates": "30g",
        "fiber": "1g",
        "sugar": "20g"
    }
})


# Nutrition info fetcher
def get_nutrition_info(food_items):
    data = []
    for item in food_items:
        item_cap = item.title()
        if item_cap in nutrition_data:
            nutrients = nutrition_data[item_cap]
            row = {"Food Item": item_cap}
            row.update(nutrients)
            data.append(row)
    return pd.DataFrame(data)

# Dietary recommendations
def generate_dietary_recommendation(food_name):
    food_name = food_name.lower()
    tips = {
        "ice cream": [
            "Limit intake due to high sugar and fat content.",
            "Opt for low-fat or frozen yogurt as alternatives.",
            "Pair with fruits for added fiber and reduced sugar spike."
        ],
        "pizza": [
            "Choose whole wheat base and load with veggies.",
            "Limit cheese and processed meats.",
            "Balance with a side salad."
        ],
        "burger": [
            "Go for grilled patties instead of fried.",
            "Use whole grain buns and add veggies.",
            "Avoid sugary soft drinks alongside."
        ],
        "chicken": [
            "Prefer grilled or baked over fried.",
            "Trim skin to reduce fat intake.",
            "Pair with fiber-rich vegetables."
        ]
    }

    return tips.get(food_name, [
        "Eat in moderation if you're watching calories.",
        "Pair with fiber-rich foods for balance.",
        "Check ingredients if you have dietary restrictions."
    ])


# Camera Input or File Upload
def camera_capture():
    st.subheader("📷 Capture Image from Camera or Upload")
    option = st.radio("Choose Image Input Method", ["Camera", "Upload"])

    if option == "Camera":
        captured_image = st.camera_input("Take a picture")
        if captured_image is not None:
            return Image.open(captured_image)
    else:
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            return Image.open(uploaded_image)
    return None


# Detect Food
def detect_food(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = keras_image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    decoded_preds = decode_predictions(predictions, top=5)[0]

    custom_labels = list(nutrition_data.keys())
    results = []
    for (_, label, confidence) in decoded_preds:
        formatted = label.replace("_", " ").title()
        if formatted in custom_labels or confidence > 0.1:
            results.append(formatted)
    return list(dict.fromkeys(results))  # remove duplicates



# Food Detection Page
import pandas as pd

# Define nutrition data
nutrition_database = {
    "banana": {"Calories": 105, "Protein (g)": 1.3, "Fat (g)": 0.3, "Carbs (g)": 27},
    "apple": {"Calories": 95, "Protein (g)": 0.5, "Fat (g)": 0.3, "Carbs (g)": 25},
    "pizza": {"Calories": 285, "Protein (g)": 12, "Fat (g)": 10, "Carbs (g)": 36},
    "burger": {"Calories": 354, "Protein (g)": 17, "Fat (g)": 17, "Carbs (g)": 29},
}

# Nutrition info fetcher
def get_nutrition_info(detected_foods):
    data = []
    for food in detected_foods:
        if food.lower() in nutrition_database:
            info = nutrition_database[food.lower()]
            data.append({"Food": food.title(), **info})
    return pd.DataFrame(data) if data else None

# Main detection page
def food_detection_page():
    """Main page for food detection and nutrition analysis."""
    st.title("🍲 Food Detection & Nutrition Analysis")

    image = camera_capture()

    if image is not None:
        st.image(image, caption="📸 Captured Image", use_column_width=True)

        # Detect food items
        detected_foods = detect_food(image)

        if detected_foods:
            st.success("🍽️ Food Detected: " + ", ".join(detected_foods))

            # Display nutritional information
            nutrition_info = get_nutrition_info(detected_foods)
            if nutrition_info is not None and not nutrition_info.empty:
                st.subheader("📊 Nutritional Information")
                st.table(nutrition_info)
            else:
                st.warning("⚠️ No nutrition data found.")

            # Provide dietary recommendations
            st.subheader("💡 Dietary Recommendations")
            for food in detected_foods:
                st.markdown(f"**{food}**:")
                st.markdown("- Eat in moderation if you're watching your calories.")
                st.markdown("- Consider alternatives if you're on a specific diet (e.g., low-carb, vegan).")
        else:
            st.error("❌ No recognizable food items found. Please try another image.")

meal_plans = {
    "Veg": {
        "Weight Loss": [
            {
                "Breakfast": "Oats with fruits",
                "Lunch": "Brown rice + Dal + Veggies",
                "Dinner": "Quinoa salad + Soup"
            },
            {
                "Breakfast": "Poha + Buttermilk",
                "Lunch": "Multigrain roti + Mixed veg + Curd",
                "Dinner": "Moong soup + Steamed broccoli"
            }
        ],
        "Weight Gain": [
            {
                "Breakfast": "Paratha + Curd + Banana",
                "Lunch": "Rice + Paneer curry + mix veg",
                "Dinner": "Khichdi + Ghee + Salad"
            }
        ],
        "Maintenance": [
            {
                "Breakfast": "Idli + Coconut chutney",
                "Lunch": "Rice + Sambar + Veg stir-fry",
                "Dinner": "Roti + Bhindi + Dal"
            }
        ]
    },
    "Non-Veg": {
        "Weight Loss": [
            {
                "Breakfast": "Boiled eggs + Fruit salad",
                "Lunch": "Grilled chicken + Quinoa + Veg",
                "Dinner": "Chicken soup + Salad"
            }
        ],
        "Weight Gain": [
            {
                "Breakfast": "Egg paratha + Milk",
                "Lunch": "Chicken curry + Rice + Veg",
                "Dinner": "Fish + Sweet potato + Salad"
            }
        ],
        "Maintenance": [
            {
                "Breakfast": "Omelette + Toast",
                "Lunch": "Fish curry + Rice + Veg",
                "Dinner": "Chicken stew + Roti"
            }
        ]
    }
}

# Sample nutrition DB (update or expand as needed)
nutrition_db = {
    "Oats with fruits": [300, 8, 5, 45],
    "Brown rice": [200, 5, 2, 45],
    "Dal": [150, 10, 1, 20],
    "Veggies": [100, 3, 1, 10],
    "Quinoa salad": [250, 8, 4, 40],
    "Soup": [120, 5, 2, 15],
    "Poha": [250, 4, 7, 35],
    "Buttermilk": [50, 3, 1, 4],
    "Multigrain roti": [100, 3, 2, 20],
    "Mixed veg": [150, 3, 4, 20],
    "Curd": [100, 5, 4, 6],
    "Moong soup": [180, 12, 3, 20],
    "Steamed broccoli": [50, 4, 1, 8],
    "Paratha": [300, 6, 10, 40],
    "Banana": [100, 1, 0, 23],
    "Paneer curry": [280, 15, 20, 10],
    "Khichdi": [320, 10, 6, 40],
    "Ghee": [90, 0, 10, 0],
    "Idli": [200, 4, 2, 35],
    "Coconut chutney": [120, 2, 12, 3],
    "Sambar": [150, 6, 2, 20],
    "Bhindi": [100, 2, 5, 10],
    "Boiled eggs": [150, 12, 10, 1],
    "Fruit salad": [100, 1, 1, 25],
    "Grilled chicken": [250, 25, 10, 0],
    "Chicken soup": [180, 15, 7, 5],
    "Egg paratha": [350, 10, 12, 40],
    "Milk": [120, 6, 5, 12],
    "Chicken curry": [300, 20, 15, 10],
    "Fish": [220, 22, 12, 0],
    "Sweet potato": [200, 2, 0, 40],
    "Omelette": [180, 10, 14, 1],
    "Toast": [120, 4, 1, 20],
    "Fish curry": [280, 18, 14, 8],
    "Chicken stew": [250, 18, 12, 10],
    "Roti": [90, 2, 1, 18]
}

# Helper to rotate meal plans
def get_rotated_meal_plan(plans, user_id):
    index = hash(user_id) % len(plans)
    return plans[index]

# Full dietary management function
def dietary_management():
    st.title("🍽️ Dietary Management System")

    st.markdown("Personalize your meal plans based on your health goals, food preferences, and allergies.")

    # User inputs
    age = st.slider("Select your age", 10, 80, 25)
    goal = st.selectbox("Select your goal", ["Weight Loss", "Weight Gain", "Maintain"])
    food_pref = st.selectbox("Food Preference", ["Veg", "Non-Veg"])
    allergies_input = st.text_input("List any allergies (comma-separated, e.g., peanuts, milk)").lower().split(",")
    activity_level = st.selectbox("Activity Level", ["Sedentary", "Active", "Very Active"])

    user_id = f"user_{age}_{goal}_{food_pref}"  # simple ID for meal plan rotation

    # Set target calories based on goal + activity
    if activity_level == "Sedentary":
        base_cal = 1800
    elif activity_level == "Active":
        base_cal = 2200
    else:
        base_cal = 2500

    if goal == "Weight Loss":
        calories = base_cal - 300
    elif goal == "Weight Gain":
        calories = base_cal + 300
    else:
        calories = base_cal

    # Get all plans
    plans = meal_plans[food_pref][goal]

    # Filter for allergies
    filtered_plans = []
    for plan in plans:
        all_dishes = " ".join(plan.values()).lower()
        if not any(allergy.strip() in all_dishes for allergy in allergies_input if allergy.strip()):
            filtered_plans.append(plan)

    # Handle no results
    if not filtered_plans:
        st.error("⚠️ No matching meal plans found based on your allergy preferences. Try fewer restrictions.")
        return

    # Rotate & display meal plan
    selected_plan = get_rotated_meal_plan(filtered_plans, user_id)

    st.success(f"🍱 Recommended {food_pref} meal plan for **{goal}** ({calories} kcal target):")

    for meal_time, dish in selected_plan.items():
        st.markdown(f"**{meal_time}:** {dish}")

    # Nutrition summary
    st.subheader("🔍 Approximate Nutrition Summary")

    total_cals, total_protein, total_fat, total_carbs = 0, 0, 0, 0
    for dish in selected_plan.values():
        for item in dish.split(" + "):
            item = item.strip()
            if item in nutrition_db:
                cals, prot, fat, carbs = nutrition_db[item]
                total_cals += cals
                total_protein += prot
                total_fat += fat
                total_carbs += carbs

    st.write(f"**Total Calories:** {total_cals} kcal")
    st.write(f"**Protein:** {total_protein} g | **Fat:** {total_fat} g | **Carbs:** {total_carbs} g")

    # Optional: Download option
    st.subheader("📥 Download Plan")
    def get_download_link(plan_dict):
        plan_text = "\n".join([f"{k}: {v}" for k, v in plan_dict.items()])
        b64 = base64.b64encode(plan_text.encode()).decode()
        return f'<a href="data:file/txt;base64,{b64}" download="meal_plan.txt">Download Meal Plan</a>'

    st.markdown(get_download_link(selected_plan), unsafe_allow_html=True)

def custom_food_recommendation():
    st.subheader("🍲 Custom Food Recommendation")

    food_pref = st.selectbox("Select your dietary preference", ["Select", "Vegetarian", "Non-Vegetarian", "Vegan"])
    meal_type = st.selectbox("Select a meal type", ["Select", "Breakfast", "Lunch", "Dinner", "Snack"])

    indian_food_database = {
        "Vegetarian": {
            "Breakfast": ["Aloo Paratha", "Upma", "Poha", "Idli Sambar", "Dhokla"],
            "Lunch": ["Paneer Tikka", "Rajma Chawal", "Chole Bhature", "Kadhi Chawal", "Mix Veg Curry"],
            "Dinner": ["Veg Biryani", "Palak Paneer", "Dal Makhani", "Malai Kofta", "Stuffed Capsicum"],
            "Snack": ["Samosa", "Chaat", "Pakora", "Bhel Puri", "Khaman"]
        },
        "Non-Vegetarian": {
            "Breakfast": ["Egg Bhurji", "Chicken Sandwich", "Masala Omelette"],
            "Lunch": ["Chicken Biryani", "Fish Curry", "Mutton Rogan Josh"],
            "Dinner": ["Tandoori Chicken", "Grilled Fish", "Chicken Stew"],
            "Snack": ["Chicken Kebab", "Egg Roll", "Chicken Pakora"]
        },
        "Vegan": {
            "Breakfast": ["Oats Porridge", "Tofu Scramble", "Vegan Poha"],
            "Lunch": ["Chickpea Curry", "Lentil Soup", "Vegan Biryani"],
            "Dinner": ["Pumpkin Soup", "Tofu Stir Fry", "Millet Khichdi"],
            "Snack": ["Roasted Chickpeas", "Fruit Salad", "Vegan Samosa"]
        }
    }

    if food_pref != "Select" and meal_type != "Select":
        recs = indian_food_database.get(food_pref, {}).get(meal_type, [])
        if recs:
            st.write(f"### Recommended {meal_type} options for {food_pref} preference:")
            for food in recs:
                st.write(f"- {food}")
        else:
            st.warning("No recommendations found for this selection.")


    
# Contact Us / Feedback Page
def contact_us():
    st.title("📞 Contact Us / Feedback")

    st.markdown("""
    We would love to hear your feedback!  
    Please let us know your thoughts, suggestions, or any issues you're facing.
    """)

    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    feedback = st.text_area("Your Feedback")

    if st.button("Submit Feedback"):
        if name and email and feedback:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    feedback TEXT NOT NULL
                )
            """)
            cursor.execute("INSERT INTO feedback (name, email, feedback) VALUES (?, ?, ?)", (name, email, feedback))
            conn.commit()
            st.success("✅ Thank you for your feedback!")
        else:
            st.error("Please fill in all the fields.")
            


# Page Routing
if selected_option == "Home":
    home()
elif selected_option == "Login":
    login()
elif selected_option == "Register":
    register()
elif selected_option == "Food Detection":
    if not st.session_state.get("logged_in"):
        st.warning("🔒 Please login to access the Food Detection page.")
    else:
        # Call your food detection function here
        food_detection_page()
elif selected_option == "Dietary Management":
    if not st.session_state.get("logged_in"):
        st.warning("🔒 Please login to access the Dietary Management page.")
    else:
        # Call your dietary management function here
        dietary_management()
        custom_food_recommendation()
elif selected_option == "Contact Us":
    contact_us()




# Footer
st.markdown("<br><p style='text-align: center;'>Powered by AI-based Food Recognition</p>", unsafe_allow_html=True)