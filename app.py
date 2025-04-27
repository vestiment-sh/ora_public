import streamlit as st
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import base64
import pandas as pd
import altair as alt
import io

im = Image.open("/Users/stevensmith/Desktop/Projects/ora_a_teachers_friend/images/logo.png")
st.set_page_config(page_title="Ora: A Teacher's Friend For The Now", page_icon = im)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set background
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background
set_background("/Users/stevensmith/Desktop/Projects/ora_a_teachers_friend/images/ora_back.png")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lobster&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Bytesized&display=swap');

    /* Style the title */
    h1 {{
        font-family: 'Lobster', cursive;
        text-align: center;
        color: white;
        font-size: 100px;
        margin-top: 20px;
    }}

    /* Style the tagline */
    .tagline {{
        font-family: 'Bytesized', sans-serif;
        text-align: center;
        color: white;
        font-size: 20px;
        margin-top: -10px;
    }}
    </style>
    <h1>Ora</h1>
    <div class="tagline">A Teacher's Friend For the Now</div>
    <br>
    """,
    unsafe_allow_html=True
)

# Load your fine-tuned model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Path to trained_model")
    model = AutoModelForSequenceClassification.from_pretrained("Path to trained_model")
    return tokenizer, model

tokenizer, model = load_model()

# Prediction function
def predict_observation(input_text):
    if not input_text.strip():
        return " "
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = logits.argmax(dim=-1).item()
    labels = ["Unsatisfactory", "Basic", "Proficient", "Accomplished"]  # Adjust these based on your model's classes
    return labels[predicted_class]

col1, spacer, col2, spacer2, col3 = st.columns([8, 2, 5, 3, 5])

with col1:
    # Create 13 text input boxes and corresponding prediction boxes
    st.header("Enter Evidence")
    inputs = []
    for i in range(1, 14):
        input_text = st.text_area(f"Evidence {i}", "", height=100)
        inputs.append(input_text)

predicted_scores = []

with col2:
    # Display predictions with editable text areas
    st.header("Observations")
    updated_predictions = []
    score_mapping = {
        "Unsatisfactory": 1,
        "Basic": 2,
        "Proficient": 3,
        "Accomplished": 4
    }
    reverse_score_mapping = {v: k for k, v in score_mapping.items()}  # Reverse mapping for validation

    for i, input_text in enumerate(inputs, 1):
        prediction = predict_observation(input_text)
        
        # Allow manual adjustment of predictions
        updated_prediction = st.text_area(f"Observation {i}", prediction, height=100)
        updated_predictions.append(updated_prediction)
        
        # Map the updated prediction to a numerical score
        predicted_scores.append(score_mapping.get(updated_prediction.strip(), 0))  # Default to 0 if invalid

with col3:
    # Calculate and display the average score
    st.header("Average")
    if predicted_scores:
        valid_scores = [score for score in predicted_scores if score > 0]  # Exclude invalid scores (0)
        if valid_scores:
            average_score = sum(valid_scores) / len(valid_scores)
            st.header(f"{average_score:.2f}")
        else:
            st.header("N/A")

# Add a horizontal line for separation
# Add a horizontal line for separation
st.markdown("---")

# Center the "Score Distribution" header with Bytesized font
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bytesized&display=swap');
    .score-distribution-header {
        font-family: 'Bytesized', sans-serif;
        text-align: center;
        color: white;
        font-size: 40px;
        margin-bottom: 20px;
    }
    </style>
    <div class="score-distribution-header">Score Distribution</div>
    """,
    unsafe_allow_html=True
)

if predicted_scores:
    valid_scores = [score for score in predicted_scores if score > 0]  # Exclude invalid scores (0)
    evidence_numbers = list(range(1, len(valid_scores) + 1))  # Evidence numbers (1 to 13)

    # Create a DataFrame for the scatter chart
    data = pd.DataFrame({
        "Evidence Number": evidence_numbers,
        "Score": valid_scores
    })

    # Define a color scale for the scores
    color_scale = alt.Scale(
        domain=[1, 2, 3, 4],
        range=["red", "yellow", "blue", "green"]
    )

    # Create a scatter chart using Altair
    scatter_chart = alt.Chart(data).mark_circle(size=100).encode(
        x=alt.X("Evidence Number:Q", scale=alt.Scale(domain=[1, 13]), title="Evidence Number"),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[1, 4]), title="Score"),
        color=alt.Color("Score:Q", scale=color_scale, legend=None),  # Map scores to colors
        tooltip=["Evidence Number", "Score"]
    )

    # Add a line of best fit
    regression_line = alt.Chart(data).transform_regression(
        "Evidence Number", "Score", method="linear"
    ).mark_line(color="red").encode(
        x="Evidence Number:Q",
        y="Score:Q"
    )

    # Combine the scatter chart and the regression line
    combined_chart = scatter_chart + regression_line

    # Set chart properties
    combined_chart = combined_chart.properties(
        width=700,  # Set the width of the chart
        height=400  # Set the height of the chart
    )

    # Display the chart
    st.altair_chart(combined_chart, use_container_width=True)

# Prepare data for download
if predicted_scores:
    valid_scores = [score for score in predicted_scores if score > 0]  # Exclude invalid scores (0)
    evidence_columns = [f"Evidence {i}" for i in range(1, 14)]  # Evidence column names
    observation_columns = [f"Observation {i}" for i in range(1, 14)]  # Observation column names

    # Create a dictionary for the DataFrame
    data = {
        **{evidence_columns[i]: [inputs[i]] for i in range(len(inputs))},  # Evidence values
        **{observation_columns[i]: [updated_predictions[i]] for i in range(len(updated_predictions))},  # Observations
    }

    # Add the average score
    if valid_scores:
        average_score = sum(valid_scores) / len(valid_scores)
        data["Average"] = [f"{average_score:.2f}"]
    else:
        data["Average"] = [""]

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Convert DataFrame to CSV
    csv = df.to_csv(index=False)

    # Add a download button
    st.download_button(
        label="Download Scores as CSV",
        data=csv,
        file_name="scores_and_average.csv",
        mime="text/csv",
    )