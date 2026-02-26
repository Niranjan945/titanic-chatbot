import streamlit as st
import requests
import base64
import json

# Must be the first Streamlit command
st.set_page_config(
    page_title="Titanic Chat Agent",
    page_icon="🚢",
    layout="wide"
)

API_URL = "https://titanic-chatbot-hi26.onrender.com"

st.title("🚢 Titanic Data Chatbot")
st.write("Ask questions about the Titanic dataset in natural language and get text + visual insights.")

# Sidebar: info + example questions
st.sidebar.header("📌 How to use")
st.sidebar.write("Type a question about the Titanic passengers and press **Ask**.")
st.sidebar.subheader("Example questions")
st.sidebar.write("- What percentage of passengers were male?\n"
                 "- Show me a histogram of passenger ages.\n"
                 "- What was the average ticket fare?\n"
                 "- How many passengers embarked from each port?\n"
                 "- What percentage of passengers survived in each class?")

# Backend health check
@st.cache_data(show_spinner=False)
def get_api_info():
    try:
        res = requests.get(f"{API_URL}/")
        if res.status_code == 200:
            return res.json()
    except Exception:
        return None

api_info = get_api_info()
if not api_info:
    st.error("❌ Cannot reach backend API. Make sure `python api.py` is running.")
else:
    st.success("✅ Backend API is online.")

# Dataset info expander
with st.expander("📊 Dataset info"):
    try:
        info_res = requests.get(f"{API_URL}/info")
        if info_res.status_code == 200:
            info = info_res.json()
            st.write(f"**Total passengers:** {info['total_passengers']}")
            st.write(f"**Columns:** {', '.join(info['columns'])}")
            st.write("**Sample rows:**")
            st.json(info["sample_data"])
        else:
            st.write("Could not fetch dataset info.")
    except Exception as e:
        st.write(f"Error fetching dataset info: {e}")

st.markdown("---")

# Main chat interface
st.subheader("💬 Ask a question")

question = st.text_input(
    "Enter your question about the Titanic dataset:",
    placeholder="e.g., What percentage of passengers were male? Show me a chart."
)

col1, col2 = st.columns([1, 3])

with col1:
    ask_btn = st.button("Ask", type="primary")

with col2:
    show_raw = st.checkbox("Show raw API response", value=False)

if ask_btn:
    if not question.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    f"{API_URL}/query",
                    json={"question": question},
                    timeout=120
                )
                if res.status_code != 200:
                    st.error(f"API error: {res.status_code} - {res.text}")
                else:
                    data = res.json()
                    if show_raw:
                        st.code(json.dumps(data, indent=2), language="json")

                    if data.get("error"):
                        st.error(f"Agent error: {data['error']}")
                    else:
                        st.markdown("#### 🧠 Answer")
                        st.write(data.get("answer", "No answer returned."))

                        viz_b64 = data.get("visualization")
                        if viz_b64:
                            st.markdown("#### 📊 Visualization")
                            img_bytes = base64.b64decode(viz_b64)
                            # The deprecated parameter is now fixed right here:
                            st.image(img_bytes, use_container_width=True)
                        else:
                            st.info("No visualization generated for this question. Try including words like 'histogram', 'chart', or 'show' in your question.")
            except Exception as e:
                st.error(f"Request failed: {e}")