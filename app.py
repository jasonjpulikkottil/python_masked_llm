import streamlit as st
from transformers import pipeline

# Streamlit App Title
st.title("മലയാളം വാചക പൂർത്തീകരണം")

# Cache the model to prevent reloading
@st.cache_resource
def load_model():
    return pipeline("fill-mask", model="ai4bharat/IndicBERTv2-MLM-only", device="cpu")

pipe = load_model()

# Example input prompt
example_text = "മലയാളം ഒരു * ഭാഷ ആണ്"

# Input field for the masked prompt
prompt = st.text_area("നിങ്ങളുടെ വാചകം മലയാളത്തിൽ ടൈപ്പ് ചെയ്യുക, വിട്ടുപോയ പദത്തിന് പകരം '*' ഉപയോഗിക്കുക.", example_text)

# Convert user input to expected MLM format (replace * with <mask>)
converted_prompt = prompt.replace("*", "<mask>")

# Button to generate text (complete the masked input)
if st.button("Generate Text"):
    if "*" in prompt:  # Ensure user has included a mask
        with st.spinner("Generating text..."):
            # Generate predictions for the masked prompt
            results = pipe(converted_prompt)
            st.subheader("Generated Text:")
            # Display the top prediction
            st.write(f"Original prompt: {prompt}")
            st.write(f"Predicted completion: {results[0]['sequence'].replace('<mask>', '*')}")
    else:
        st.warning("വിട്ടുപോയ പദത്തിന്  '*' ഉപയോഗിക്കുക.")
