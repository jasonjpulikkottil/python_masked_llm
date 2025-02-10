import streamlit as st
from transformers import pipeline

st.title("മലയാളം വാചക പൂർത്തീകരണം")

@st.cache_resource
def load_model():
    return pipeline("fill-mask", model="ai4bharat/IndicBERTv2-MLM-only", device="cpu")

pipe = load_model()

example_text = "മലയാളം ഒരു * ഭാഷ ആണ്"

prompt = st.text_area("നിങ്ങളുടെ വാചകം മലയാളത്തിൽ ടൈപ്പ് ചെയ്യുക, വിട്ടുപോയ പദത്തിന് പകരം '*' ഉപയോഗിക്കുക.", example_text)
prompt = prompt.replace("*", "[MASK]")

if st.button("സൃഷ്ടിക്കുക"):
    if "[MASK]" in prompt:  # Ensure user has included a mask
        with st.spinner("ഫലം സൃഷ്ടിക്കപ്പെടുന്നു..."):
            # Generate predictions for the masked prompt
            results = pipe(prompt, top_k=3)  # Get top 3 predictions
            st.subheader("സൃഷ്ടിച്ച വാചകം:")

            # Display the top 3 predictions
            for i, result in enumerate(results[:3]):
                completed_sentence = result["sequence"].replace("[MASK]", "*")
                st.write(f"#{i+1}: {completed_sentence}")

    else:
        st.warning("വിട്ടുപോയ പദത്തിന് പകരം * ഉപയോഗിക്കുക.")
