import streamlit as st

MODEL_FAMILIES = {
    "Supervised": {
        "Number (regression)": ["Linear Regression", "Decision Trees", "Random Forest", "Gradient Boosting"],
        "Label (classification)": ["Logistic Regression", "Decision Trees", "Random Forest", "Support Vector Machines", "k-Nearest Neighbors", "Naive Bayes"]
    },
    #"Supervised": [
    #    "Linear Regression",
    #    "Logistic Regression",
    #    "Decision Trees",
    #    "Random Forest",
    #    "Gradient Boosting",
    #    "Support Vector Machines",
    #    "k-Nearest Neighbors",
    #    "Naive Bayes",
    #],
    "Unsupervised": [
        "k-Means",
        "Hierarchical Clustering",
        "DBSCAN",
        "Gaussian Mixture Models",
        "PCA",
        "t-SNE",
        "UMAP",
    ],
    "NLP": [
        "Bag of Words",
        "TF-IDF",
        "Word2Vec",
        "GloVe",
        "RNNs",
        "LSTM/GRU",
        "Transformers",
    ],
    "Deep Learning": [
        "Feedforward Networks",
        "CNNs",
        "Autoencoders",
        "GANs",
        "Vision Transformers",
    ],
}


def slug(value: str) -> str:
    return value.replace("-", " ").replace("/", " ").replace(" ", "_")


def family_overview_path(family: str) -> str:
    return f"pages/{slug(family)}_Overview.py"


def model_page_path(family: str, model: str) -> str:
    return f"pages/{slug(family)}_{slug(model)}.py"


def safe_switch_page(path: str) -> None:
    if not hasattr(st, "switch_page"):
        st.write("Navigation requires Streamlit pages support.")
        return

    try:
        st.switch_page(path)
        return
    except Exception:
        if path.startswith("pages/"):
            st.switch_page(path[len("pages/"):])
            return
        raise


def handle_family_change() -> None:
    st.session_state["nav_model"] = "Overview"


def handle_model_change(family: str) -> None:
    selection = st.session_state.get("nav_model")
    if selection is None:
        return
    if selection == "Overview":
        safe_switch_page(family_overview_path(family))
        return
    safe_switch_page(model_page_path(family, selection))

'''
def render_sidebar() -> None:    
    with st.sidebar:
        st.header("Model Families")
        if st.button("About / Home", key="nav_home"):
            safe_switch_page("main.py")

        families = list(MODEL_FAMILIES.keys())
        current_family = st.session_state.get("nav_family", families[0])
        if current_family not in MODEL_FAMILIES:
            current_family = families[0]

        st.selectbox(
            "Model family",
            options=families,
            key="nav_family",
            index=families.index(current_family),
            on_change=handle_family_change,
        )

        family_models = MODEL_FAMILIES[current_family]
        model_options = ["Overview", *family_models]
        current_model = st.session_state.get("nav_model", "Overview")
        if current_model not in model_options:
            current_model = "Overview"

        st.selectbox(
            "Model",
            options=model_options,
            key="nav_model",
            index=model_options.index(current_model),
            on_change=handle_model_change,
            kwargs={"family": current_family},
        )
        '''

def render_sidebar():
    with st.sidebar:
        st.header('Model Family')
        families = list(MODEL_FAMILIES.keys())

        model_family = st.selectbox("Model family", options=families, key="nav_family", index=None, placeholder="Make a selection")

        # Initialize an empty list for models
        model_options = []
        
        if model_family == None:
            return "no selection"
        
        elif model_family == 'Supervised':
            problem_type = st.selectbox(
                "Predicting a number or label?",
                options=["Number (regression)", "Label (classification)"],
                key="problem_type"
            )
            # Filter models based on the selected sub-type
            model_options = MODEL_FAMILIES["Supervised"][problem_type]
        
        else:
            # For Unsupervised, NLP, etc., just grab the list directly
            model_options = MODEL_FAMILIES[model_family]

        # The "Model" selectbox now uses the filtered list
        chosen_model = st.selectbox(
            "Model",
            options=["Overview"] + model_options,
            key="nav_model"
        )
        return "selection made"