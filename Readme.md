# 🚢 Titanic Data Chatbot

A full-stack AI application that allows users to ask natural language questions about the Titanic dataset and receive both text answers and dynamically generated charts. 

Powered by **LangChain**, **Groq (Llama 3)**, **FastAPI**, and **Streamlit**.

## 🌟 Features
* **Natural Language Querying:** Ask questions like "What was the average ticket fare?" or "What percentage of passengers were male?"
* **Dynamic Visualizations:** The AI writes Python code on the fly to generate Matplotlib/Seaborn charts and sends them to the frontend via Base64 encoding.
* **FastAPI Backend:** A robust, decoupled REST API that manages the LangChain DataFrame Agent.
* **Streamlit Frontend:** A clean, interactive web interface for users to chat with the data.

## 🛠️ Prerequisites
* Python 3.9+
* A free [Groq API Key](https://console.groq.com/keys)

## 🚀 Setup Instructions

**1. Clone the repository and navigate to the project folder:**
```bash
git clone [https://github.com/Niranjan945/titanic-chatbot.git](https://github.com/Niranjan945/titanic-chatbot.git)
cd titanic-chatbot
