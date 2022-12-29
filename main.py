import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.experimental_memo
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained('google/pegasus-xsum')
    return model

@st.experimental_memo
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('google/pegasus-xsum')
    return tokenizer


model = load_model()
tokenizer = load_tokenizer()

uploaded_file = st.file_uploader("Choose a file", type=[".xlsx", "xls"])
if uploaded_file:
    print(uploaded_file.__dict__)
    df = pd.read_excel(uploaded_file)
    st.dataframe(df)
    feedbacks = []
    for col in df:
        c = ""
        for row in df[col]:
            r = row.strip()
            if not (r.endswith(".") or r.endswith("?") or r.endswith("!")):
                r += ". "
            c += r
        feedbacks.append(c.strip())
            
    # st.write(feedbacks)
    summaries = []

    for i, feedback in enumerate(feedbacks):
        print(len(feedback))
        tokens_input = tokenizer.encode("summarize:" + feedback, return_tensors='pt', max_length=512, truncation=True)
        ids = model.generate(tokens_input, min_length=len(feedback) // 4, max_length=120)
        summary = tokenizer.decode(ids[0], skip_special_tokens=True)

        st.write(feedback)
        print(len(summary), summary)

        st.write(f"Question {i+1}: " + summary)

