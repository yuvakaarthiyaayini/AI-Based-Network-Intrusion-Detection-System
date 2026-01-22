import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


st.set_page_config(
    page_title="AI-Based NIDS",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center; color: #4B7BEC;'>AI-Based Network Intrusion Detection System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray;'>Detect malicious network traffic using Machine Learning</p>",
    unsafe_allow_html=True
)
st.markdown("---")


def generate_data(samples=2000):
    np.random.seed(42)
    data = {
        "duration": np.random.randint(0, 1000, samples),
        "src_bytes": np.random.randint(0, 50000, samples),
        "dst_bytes": np.random.randint(0, 50000, samples),
        "count": np.random.randint(0, 100, samples),
        "srv_count": np.random.randint(0, 100, samples),
        "label": np.random.choice([0, 1], samples, p=[0.7, 0.3])
    }
    return pd.DataFrame(data)


df = generate_data()
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


st.sidebar.header("Model Controls")
n_estimators = st.sidebar.slider("Number of Trees", 10, 500, 100, step=10)
test_size = st.sidebar.slider("Test Data Size (%)", 10, 50, 25, step=5)

train_model = st.sidebar.button("Train Model Now")


with st.container():
    st.subheader("Model Training")

    if train_model:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model Trained Successfully | Accuracy: {accuracy * 100:.2f}%")

        cm = confusion_matrix(y_test, y_pred)
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.session_state["model"] = model
    else:
        st.info("Train the model using the sidebar to enable live traffic analysis.")


st.markdown("---")
st.subheader("Live Network Traffic Test")

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        duration = st.number_input("Duration", 0, 10000, 200)
        src_bytes = st.number_input("Source Bytes", 0, 100000, 5000)

    with col2:
        dst_bytes = st.number_input("Destination Bytes", 0, 100000, 3000)
        count = st.number_input("Connection Count", 0, 200, 10)

    with col3:
        srv_count = st.number_input("Service Count", 0, 200, 5)

    if st.button("Analyze Traffic"):
        if "model" not in st.session_state:
            st.error("Please train the model first")
        else:
            input_data = np.array([[duration, src_bytes, dst_bytes, count, srv_count]])
            prediction = st.session_state["model"].predict(input_data)

            if prediction[0] == 1:
                st.error("Intrusion Detected (Malicious Traffic)")
            else:
                st.success("Normal Network Traffic")


st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Developed using Python | Machine Learning | Streamlit</p>",
    unsafe_allow_html=True
)
