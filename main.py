import streamlit as st

from sidebar import render_sidebar, slug

from functions.data_prep.import_data import data_loader

from model_pages import Supervised_Linear_Regression

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


dict_selected_model_page = {
    "Supervised_Linear_Regression": Supervised_Linear_Regression
}

if selection_status != "no selection":
    family = st.session_state.get("nav_family")
    model = st.session_state.get("nav_model")
    if family and model:
        page_key = f"{slug(family)}_{slug(model)}"
        if page_key in dict_selected_model_page:
            dict_selected_model_page[page_key].app()
