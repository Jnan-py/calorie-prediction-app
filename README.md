# 🔥 Calorie Prediction & Visualization App

This is a Streamlit-based web application that predicts the number of calories burned based on user input parameters. The app also provides interactive data visualizations to understand relationships between various features and calories burned.

## 🚀 Features

- **Calorie Prediction**: Uses machine learning models to predict calories burned.
- **Interactive Visualizations**: Displays scatter plots and a correlation heatmap for insights.
- **Model Training**: Trains a regression model on the dataset and saves the best model.
- **User-Friendly UI**: Built using Streamlit with an intuitive layout.

## 📦 Installation

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/Jnan-py/calorie-prediction-app.git
cd calorie-prediction-app
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)

```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 3️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### Running the Streamlit App

```sh
streamlit run app.py
```

This will open the web app in your browser.

## 📂 Project Structure

```
calorie-prediction-app/
│── Dataset/
│   ├── calories_burnt.csv
│── app.py
│── requirements.txt
│── README.md
│── calories_model.pkl  (Generated after training)
│── scaler.pkl          (Generated after training)
│── .gitignore
```

## 🔧 Model Training & Prediction

- The app trains multiple models (Linear Regression, XGBoost, Lasso, Random Forest, Ridge) and selects the best-performing one.
- The trained model and scaler are saved locally as `calories_model.pkl` and `scaler.pkl`.
- The model predicts calories based on user input features.

## 🖼 Visualizations

- **Scatter Plots**: Relationships between `Age`, `Height`, `Weight`, `Duration`, `Heart Rate`, `Body Temperature`, and `Calories`.
- **Correlation Heatmap**: Displays feature relationships.

## 🙌 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.

---

⭐ Feel free to fork and enhance this project!
