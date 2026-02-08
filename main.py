import streamlit as st

from sidebar import render_sidebar

from functions.data_prep.import_data import data_loader

st.set_page_config(page_title="Which ML?", layout="wide")

selection_status = render_sidebar()

if selection_status == "no selection":
    st.title("Which ML?")
    with st.expander(label="About:"):
        st.write(
            "A collection of ML models, visually explained, with references to other supporting materials."
        )
        st.write(
            "Use the menu bar to navigate to different classes of models, then select a model to review."
        )
else:
    df_full, X, y = data_loader(st.session_state["problem_type"])

