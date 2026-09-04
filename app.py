"""
Smart Food Storage - Final App (Streamlit)
--------------------------------------------------
Combines all three upgraded models into one interface:
    1. Upload a food photo -> identifies the food (30-class image recognition model)
    2a. If it's apple/banana/carrot/tomato -> predicts freshness DIRECTLY from
        that same photo (photo-based freshness model)
    2b. For any other food -> falls back to entering days/storage details
        (the original Random Forest model, which covers all 30 foods)
    3. Shows real recipes matched from an 8000+ recipe dataset

HOW TO RUN THIS APP:
    1. Install requirements:
       pip install streamlit tensorflow scikit-learn joblib pandas numpy pillow --break-system-packages

    2. Put these files in the SAME folder as this script:
       - food_model.keras          (30-class food recognition)
       - class_names.txt
       - freshness_model.keras     (photo-based freshness, apple/banana/carrot/tomato)
       - freshness_class_names.txt
       - expiry_model.joblib       (Random Forest freshness, all 30 foods)
       - food_encoder.joblib
       - storage_encoder.joblib
       - status_encoder.joblib
       - recipe_lookup.json        (real recipes matched to each food)

    3. Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import json
from PIL import Image

st.set_page_config(page_title="Smart Food Storage", page_icon="🍎")
st.title("🍎 Smart Food Storage")
st.write("Upload a food photo to identify it, then check how fresh it likely is.")

# Foods the photo-based freshness model was trained on - for anything else,
# we fall back to the Random Forest model with typed-in storage details
PHOTO_FRESHNESS_FOODS = {"apple", "banana", "carrot", "tomato"}

STATUS_DISPLAY = {
    "fresh": ("✅", "Fresh"),
    "expiring_soon": ("⚠️", "Expiring Soon"),
    "spoiled": ("❌", "Spoiled"),
}


# ---------- Load everything once, cached ----------
@st.cache_resource
def load_image_model():
    model = tf.keras.models.load_model("food_model.keras")
    with open("class_names.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names


@st.cache_resource
def load_freshness_photo_model():
    model = tf.keras.models.load_model("freshness_model.keras")
    with open("freshness_class_names.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names


@st.cache_resource
def load_expiry_model():
    model = joblib.load("expiry_model.joblib")
    food_encoder = joblib.load("food_encoder.joblib")
    storage_encoder = joblib.load("storage_encoder.joblib")
    status_encoder = joblib.load("status_encoder.joblib")
    return model, food_encoder, storage_encoder, status_encoder


@st.cache_resource
def load_recipes():
    with open("recipe_lookup.json") as f:
        return json.load(f)


try:
    image_model, class_names = load_image_model()
    freshness_photo_model, freshness_class_names = load_freshness_photo_model()
    expiry_model, food_encoder, storage_encoder, status_encoder = load_expiry_model()
    recipe_lookup = load_recipes()
    models_loaded = True
except Exception as e:
    st.error(f"Couldn't load model files: {e}")
    models_loaded = False


def show_recipes(food_name, prediction):
    """Displays recipe suggestions for a food, or a discard warning if spoiled."""
    if prediction == "spoiled":
        st.warning(f"This {food_name} is likely spoiled — consider discarding it.")
        return

    recipes = recipe_lookup.get(food_name, [])
    if prediction == "expiring_soon":
        st.info(f"Your {food_name} is expiring soon — here are some recipes to use it up:")
    else:
        st.write(f"Here are some recipe ideas for your {food_name}:")

    if recipes:
        for recipe in recipes:
            with st.expander(f"🍳 {recipe['name']} (⭐ {recipe['rating']})"):
                st.write(recipe["instructions"])
                if recipe.get("prep_time") or recipe.get("cook_time"):
                    st.caption(f"Prep: {recipe.get('prep_time', '?')} | Cook: {recipe.get('cook_time', '?')}")
    else:
        st.write("No recipes found for this food yet.")


if models_loaded:
    # ---------- Step 1: Identify the food ----------
    st.header("Step 1: Identify the food")
    uploaded_photo = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png"])

    identified_food = None
    img = None

    if uploaded_photo is not None:
        img = Image.open(uploaded_photo).convert("RGB")
        st.image(img, caption="Uploaded photo", width=250)

        img_resized = img.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)

        predictions = image_model.predict(img_array)
        scores = tf.nn.softmax(predictions[0])
        identified_food = class_names[np.argmax(scores)]
        confidence = 100 * np.max(scores)

        st.success(f"Identified: **{identified_food}** ({confidence:.1f}% confidence)")

    # ---------- Step 2: Freshness ----------
    st.header("Step 2: Check freshness")

    if identified_food in PHOTO_FRESHNESS_FOODS and img is not None:
        # Use the photo-based freshness model directly on the same uploaded photo
        st.write(f"Checking freshness directly from the photo (supported for {identified_food})...")

        if st.button("Predict freshness from photo"):
            fresh_predictions = freshness_photo_model.predict(img_array)
            fresh_scores = tf.nn.softmax(fresh_predictions[0])
            prediction = freshness_class_names[np.argmax(fresh_scores)]
            confidence = 100 * np.max(fresh_scores)

            icon, label = STATUS_DISPLAY.get(prediction, ("", prediction))
            st.markdown(f"### {icon} Status: {label}")
            st.write(f"Confidence: {confidence:.1f}%")

            show_recipes(identified_food, prediction)

    else:
        # Fall back to the Random Forest model with manually entered details
        if identified_food is not None:
            st.write(f"Photo-based freshness isn't available for **{identified_food}** yet — enter storage details instead:")

        available_foods = list(food_encoder.classes_)
        default_index = available_foods.index(identified_food) if identified_food in available_foods else 0

        food_choice = st.selectbox("Food item", available_foods, index=default_index)
        days_since_purchase = st.number_input("Days since purchase", min_value=0.0, value=3.0, step=1.0)
        storage_choice = st.selectbox("Storage type", list(storage_encoder.classes_))
        temperature_c = st.slider("Temperature (°C)", min_value=0.0, max_value=35.0, value=20.0)
        humidity_level = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)

        if st.button("Predict freshness"):
            food_encoded = food_encoder.transform([food_choice])[0]
            storage_encoded = storage_encoder.transform([storage_choice])[0]

            input_data = pd.DataFrame([{
                "food_item_encoded": food_encoded,
                "days_since_purchase": days_since_purchase,
                "storage_type_encoded": storage_encoded,
                "temperature_c": temperature_c,
                "humidity_level": humidity_level,
            }])

            prediction_encoded = expiry_model.predict(input_data)[0]
            prediction = status_encoder.inverse_transform([prediction_encoded])[0]
            probabilities = expiry_model.predict_proba(input_data)[0]
            confidence = max(probabilities) * 100

            icon, label = STATUS_DISPLAY.get(prediction, ("", prediction))
            st.markdown(f"### {icon} Status: {label}")
            st.write(f"Confidence: {confidence:.1f}%")

            show_recipes(food_choice, prediction)
