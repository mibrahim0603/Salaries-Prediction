# 💰 Employee Salary Predictor (>100k)

An interactive Machine Learning web application built with **Streamlit** and **Scikit-Learn** that predicts whether an employee's salary exceeds $100,000 annually based on their company, job role, and highest academic degree.

Live App Link: *[Paste your Streamlit Cloud URL here once it is deployed!]*

---

## 🚀 Features
* **Interactive UI:** Simple dropdown menus for users to input employee characteristics.
* **Instant Predictions:** Powered by a pre-trained Decision Tree Classifier loaded dynamically via Python's `pickle` library.
* **Cached ML Assets:** Uses Streamlit's `@st.cache_resource` to keep the model and encoders loaded in memory for lightning-fast inference.

---

## 🧠 How It Works (Machine Learning Pipeline)
1. **Data Preprocessing:** Categorical features (`company`, `job`, `degree`) are converted into numerical formats using Scikit-Learn's `LabelEncoder` during training.
2. **Model Training:** A `DecisionTreeClassifier` is trained on historical data to map these features to a binary target:
   * **`1`** = Salary is greater than $100k
   * **`0`** = Salary is less than $100k
3. **Serialization:** The trained model and individual feature encoders are exported into serialized `.pkl` files (Pickle format) so the web app can use them seamlessly without retraining.

---

## 📂 Project Structure
```text
├── app.py                  # The main Streamlit web application code
├── requirements.txt        # Production dependencies for cloud deployment
├── README.md               # Project documentation
├── decision_tree_model.pkl # Saved Scikit-Learn Decision Tree Classifier
├── le_1.pkl     # Saved LabelEncoder for the 'company' feature
├── le_2.pkl         # Saved LabelEncoder for the 'job' feature
└── le_1.pkl      # Saved LabelEncoder for the 'degree' feature