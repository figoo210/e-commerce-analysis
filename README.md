# Interactive Dashboard for E-Commerce Sales Analysis  

This repository contains the code, data, and resources for analyzing e-commerce sales trends and creating an interactive dashboard. The project focuses on exploring temporal patterns, assessing the impact of external factors, and providing actionable insights through data visualization.  

## **Project Overview**  

### **Objectives:**  
1. Perform a comprehensive time series analysis of e-commerce sales data.  
2. Investigate the influence of external factors such as holidays, economic indicators, and lockdown periods.  
3. Develop an interactive dashboard for dynamic exploration of sales trends and patterns.  

### **Key Features:**  
- **Data Enrichment:** Enhanced dataset with temporal features, lagged metrics, rolling averages, holidays, and economic indicators.  
- **Time Series Decomposition:** Trend, seasonal, and residual components analyzed for each product sector.  
- **Correlation Analysis:** Identified relationships between sales and external factors such as GDP, inflation, and holidays.  
- **Statistical Testing:** T-tests validated the impact of events like lockdowns and holidays on sales.  
- **Interactive Dashboard:** Built with Streamlit to enable users to explore data dynamically and gain actionable insights.  

---

## **Technologies Used**  

### **Languages and Libraries:**  
- **Python**  
  - Pandas  
  - Numpy  
  - Statsmodels  
  - Scipy  
  - Plotly  
  - Matplotlib  
- **Streamlit**: For creating the interactive dashboard.  

### **Data Sources:**  
1. Original e-commerce dataset (2013–2024).  
2. **External Data:**  
   - Holidays: [Calendarific API](https://calendarific.com/)  
   - Economic Indicators: [ISTAT](https://www.istat.it/en/) and [World Bank Open Data](https://data.worldbank.org/)  
   - Event Data: [World Fishing Championship](https://www.worldfishingchampionship.com/), [Serie A](https://www.legaseriea.it/en), and [UEFA](https://www.uefa.com/)  
   - COVID-19 Lockdown Data: [Italian Government COVID-19 Portal](https://www.salute.gov.it/portale/home.html)  

---

## **Getting Started**  

### **Prerequisites:**  
- Python 3.8 or higher  
- Required libraries listed in `requirements.txt`  

### **Installation:**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/figoo210/e-commerce-analysis.git
   cd e-commerce-sales-analysis
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

### **Running the Dashboard:**  
1. Navigate to the dashboard directory:  
   ```bash
   cd dashboard
   ```  
2. Launch the Streamlit app:  
   ```bash
   streamlit run app.py
   ```  
3. Open the provided URL to interact with the dashboard in your browser.  

---

---

## **Key Insights**  

### **Sector-Specific Trends:**  
- The *Pesca* sector showed a steady upward trend, driven by targeted events like the World Fishing Championship.  
- The *Calcio* sector exhibited strong seasonality aligned with major sporting events.  

### **Impact of External Factors:**  
- Holidays such as Black Friday and Christmas significantly increased sales.  
- Economic conditions (GDP and inflation) showed moderate positive correlations.  
- Lockdown periods led to an average 20% decline in sales across sectors.  

### **Dashboard Features:**  
- Interactive visualizations for sector-wise trends and seasonality.  
- Correlation heatmaps to explore relationships between sales and external factors.  
- Statistical summaries to validate significant differences in sales patterns.  

---

## **Future Work**  
1. Integrate advanced predictive analytics using machine learning models (e.g., LSTM, ARIMA).  
2. Enable real-time data enrichment and visualization with tools like Apache Kafka.  
3. Expand analysis with customer segmentation and sentiment analysis for a deeper understanding of purchasing behavior.  

---

## **Resources**  
- **Colab Notebook:** [Colab Notebook Link](https://colab.research.google.com/drive/1t5mhq1JJdv_wAkIiQOVDPif8cPVn4e4b?usp=sharing)

---

## **License**  
This project is licensed under the MIT License - see the `LICENSE` file for details.  
