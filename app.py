import streamlit as st
import joblib

# Load the vectorizer and model (assuming they are in the correct directory)
# If 'vectorizer.joblib' and 'lr_model.joblib' are not found, this will raise an error.
try:
    vectorizer = joblib.load("vectorizer.joblib")
    model = joblib.load("lr_model.joblib")
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}. Please ensure 'vectorizer.joblib' and 'lr_model.joblib' are in the same directory.")
    st.stop() # Stop execution if files aren't found

# --- Custom Styling for White Background ---
# This CSS targets the main block container of the Streamlit page.
st.markdown(
    """
    <style>
    .stApp {
        background-color: white; /* Sets the main background to white */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- End of Custom Styling ---

st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is **Fake** or **Real**.")

news_input = st.text_area("News Article:", "")

if st.button("Check News"):
    if news_input.strip():
        # Transform the input text using the loaded vectorizer
        transform_input = vectorizer.transform([news_input])
        
        # Make the prediction
        prediction = model.predict(transform_input)

        st.markdown("---") # Visual separator before result
        
        # Display the result
        if prediction[0] == 1:
            st.success("The News is **Real!** ✅")
        else:
            st.error("The News is **Fake!** ❌")
            
    else:
        st.warning("Please enter some text to analyze.")