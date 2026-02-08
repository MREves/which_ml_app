import streamlit as st

from sidebar import render_sidebar

st.set_page_config(page_title="Which ML?", layout="wide")

render_sidebar()

st.title("Which ML?")
with st.expander(label="About:"):
    st.write(
        "A collection of ML models, visually explained, with references to other supporting materials."
    )
    st.write(
        "Use the menu bar to navigate to different classes of models, then select a model to review."
    )
