import streamlit as st

from sidebar import MODEL_FAMILIES, model_page_path, render_sidebar

render_sidebar()

st.title("Supervised Models")
st.write("This is the landing page for the family. Add summaries, diagrams, and guidance here.")

st.markdown("**Models in this family:**")
for model in MODEL_FAMILIES.get("Supervised", []):
    st.page_link(model_page_path("Supervised", model), label=model)
