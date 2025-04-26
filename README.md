#  Car Price Predictor

[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-blue.svg)](https://streamlit.io) [![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

A local, interactive Streamlit app that trains a simple linear-regression model—right in your browser—to estimate used-car prices from your own data. No third-party servers, no hidden APIs, just transparent, reproducible pricing.

---

##  Features

- **Data-Driven Estimates**  
  - Automatically preprocesses your CSV (Brand, model, Year, kmDriven, Transmission, FuelType, AskPrice)  
  - One-hot encodes categorical fields and fits a `LinearRegression` pipeline  

- **Instant Predictions**  
  - Streamlit sliders & dropdowns for Brand, Model, Year, Mileage, Transmission & Fuel Type  
  - Click “Predict” to see your car’s estimated market value  

- **Visual Diagnostics**  
  - Plotly Express scatter plot of Actual vs. Predicted prices  
  - R² gauge chart shows model accuracy at a glance  

- **User-Friendly Controls**  
  - Sidebar memory monitor and “Clear Cache & Retrain” button  
  - Upload your own CSV or use the bundled demo dataset  

---

##  Quick Start

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Sahil-Harchandani/Car-Price-Predictor.git
   cd Car-Price-Predictor
   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**  
   ```bash
   streamlit run car-price-prediction.py
   ```

---

##  Tech Stack

| Component            | Technology                        |
|----------------------|-----------------------------------|
| **Language**         | Python                            |
| **Data Handling**    | Pandas, NumPy                     |
| **Modeling**         | scikit-learn (LinearRegression)   |
| **UI**               | Streamlit                         |
| **Visualization**    | Plotly (Express + Graph Objects)  |
| **System Utils**     | psutil, garbage collection        |

---

##  Project Structure

```
Car-Price-Predictor/
├── car-price-prediction.py              # Main Streamlit application
├── used_car_dataset.csv                 # Demo dataset (swap in your own)
├── requirements.txt                     # Python dependencies
├── docs/
│   ├── screenshot_input.png             # Input form preview
│   └── screenshot_performance.png       # R² gauge & scatter plot
├── LICENSE                              # GPL-3.0 License
└── README.md                            # You are here!
```

---

##  Screenshots

<p float="left">
  <img src="https://github.com/user-attachments/assets/8cf4f094-361a-4e44-8ef0-219a83420908" width="45%" alt="Input Form"/>
  <img src="https://github.com/user-attachments/assets/e6212cc0-1c34-444c-9b77-a135af0b9475" width="45%" alt="Model Performance"/>
</p>

---

##  License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for full details.

---

##  Contributing

Contributions, issues, and feature requests are welcome!  
1. Fork the repo  
2. Create a branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m 'Add feature'`)  
4. Push to your branch (`git push origin feature-name`)  
5. Open a Pull Request

---

##  Credits

- **Streamlit** – effortless, interactive UIs  
- **scikit-learn** – solid linear regression modeling  
- **Pandas & NumPy** – data wrangling & math  
- **Plotly** – rich, client-side plotting  

---

⭐ If Car Price Predictor helps you nail the perfect price, don’t forget to star the repo!
