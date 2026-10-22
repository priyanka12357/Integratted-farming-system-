# 🌱 Integrated farming 

## 📌 Project Overview
The Agricultural AI  System is an AI-powered application designed to help farmers make informed agricultural decisions using machine learning and data-driven insights.

The system aims to support crop selection, disease identification, fertilizer recommendations, and weather-based agricultural planning.

## ✨ Features
- 🌾 **Crop Recommendation:** Suggests suitable crops based on agricultural input data.
- 🍃 **Plant Disease Detection:** Helps identify plant diseases using the project's implemented disease detection functionality.
- 🧪 **Fertilizer Recommendation:** Provides fertilizer suggestions based on available input parameters.
- 🌦️ **Weather Forecasting:** Supports agricultural planning with weather information.
- 🤖 **AI Chatbot:** Answers agriculture-related questions using an API-powered chatbot.
- 🗄️ **Database Integration:** Stores and manages application data.

## 🛠️ Technologies Used
- Python
- Machine Learning
- Flask (if used by the application)
- HTML, CSS, and JavaScript (if used by the interface)
- SQLite (if used by the database)
- External API integration for the chatbot and/or weather features

## 📂 Project Structure
- `app.py` — Main application entry point
- `check.py` — Supporting application or checking logic
- `config.py` — Application configuration
- `database.py` — Database operations
- `disease.py` — Disease detection functionality
- `fertilizer.py` — Fertilizer recommendation functionality
- `model.py` — Machine learning model logic
- `weather_forecast.py` — Weather forecasting functionality
- `krishimitra.db` — Database file, if required by the application

## ⚙️ Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/priyanka12357/Integrated-farming-system-.git
   ```

2. Open the project directory:

   ```bash
   cd Integrated-farming-system-
   ```

3. Create and activate a Python virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows:

   ```bash
   .venv\Scripts\activate
   ```

4. Install the dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

5. Configure the required API keys using environment variables. Do not commit real API keys or passwords to GitHub.

6. Start the application using the appropriate command for the project's framework. For a Flask application, this may be:

   ```bash
   python app.py
   ```

## 🔐 Security
- Store API keys and credentials in environment variables.
- Keep `.env` files out of version control.
- Do not publish sensitive database contents or credentials.

## 🎯 Objective
To explore how artificial intelligence and machine learning can support smarter, more informed, and efficient agricultural practices.

## 👩‍💻 Project
**Agricultural AI Decision-Support System**

Developed using Python and AI/ML technologies.
