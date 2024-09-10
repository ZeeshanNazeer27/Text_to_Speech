%%writefile audio.py
import streamlit as st

import transformers
from transformers import pipeline
import streamlit as st

st.title("Building Generative AI Tool")

st.subheader("Converting Text to Speech")

gender = st.selectbox("Choose voice type", ('Male', 'Female'))

if gender == 'Male':
    pipe = pipeline("text-to-speech", model="suno/bark-small", device="cuda")
else:
    pipe = pipeline("text-to-speech", model="suno/bark", device="cuda") 

text = st.text_input("Enter your text here...", value="")

# Generate speech only when text is provided
if text:
    output = pipe(text)
    st.audio(output["audio"], sample_rate=output["sampling_rate"])
