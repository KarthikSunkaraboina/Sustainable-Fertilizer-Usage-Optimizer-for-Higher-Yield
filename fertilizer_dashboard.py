import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.express as px
import io

# Page setup
st.set_page_config(page_title="TA Fertilizer Optimizer", layout="wide")

# Display logo and title
st.image("ta_logo.png", width=100)
st.markdown("<h1 style='text-align: center;'>TA Fertilizer Optimizer</h1>", unsafe_allow_html=True)
st.markdown("### Empowering Smart Agriculture with Data")

# Sidebar
with st.sidebar:
    st.image("ta_logo.png", width=120)
    st.markdown("## TA Fertilizer Optimizer")
    st.markdown("Optimize fertilizer usage for higher yield and sustainability.")
    st.markdown("---")
    st.markdown("📁 Dataset: Fertilizer_dataset.csv")
    st.markdown("📞 Contact: agri-support@example.com")

# Load data and train model
data = pd.read_csv("Fertilizer_dataset.csv")
X = data[['Nitrogen','Phosphorus','Potassium','Soil_pH','Rainfall','Sunlight']]
y = data['Yield']
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)
importances = model.feature_importances_

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home", "🧪 Optimization", "📊 Visual Insights", "📥 Download Report", "📝 Feedback"
])

# Home Tab
with tab1:
    st.header(" Sustainable Fertilizer Usage Optimizer")
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.slider("Nitrogen (kg/ha)", 50, 200, 120)
        P = st.slider("Phosphorus (kg/ha)", 20, 100, 60)
    with col2:
        K = st.slider("Potassium (kg/ha)", 20, 120, 60)
        soil_pH = st.slider("Soil pH", 5.5, 7.5, 6.5)
    with col3:
        rainfall = st.slider("Rainfall (mm)", 50, 300, 150)
        sunlight = st.slider("Sunlight (hrs/day)", 4.0, 10.0, 7.0)

    target_yield = st.slider("Target Yield (tons/ha)", 5, 30, 15)

    input_df = pd.DataFrame({
        'Nitrogen':[N], 'Phosphorus':[P], 'Potassium':[K],
        'Soil_pH':[soil_pH], 'Rainfall':[rainfall], 'Sunlight':[sunlight]
    })
    predicted_yield = model.predict(input_df)[0]

    colA, colB = st.columns(2)
    with colA:
        st.metric("📊 Predicted Yield", f"{predicted_yield:.2f} tons/ha")
    with colB:
        status = "✅ Meets Target" if predicted_yield >= target_yield else "⚠️ Below Target"
        st.metric("🎯 Target Status", status)

# Optimization Tab
with tab2:
    def optimize_fertilizer(target_yield, soil_pH, rainfall, sunlight):
        best_yield = 0
        best_combo = None
        for N_opt in range(50, 201, 20):
            for P_opt in range(20, 101, 20):
                for K_opt in range(20, 121, 20):
                    X_input = pd.DataFrame({
                        'Nitrogen':[N_opt], 'Phosphorus':[P_opt], 'Potassium':[K_opt],
                        'Soil_pH':[soil_pH], 'Rainfall':[rainfall], 'Sunlight':[sunlight]
                    })
                    pred_y = model.predict(X_input)[0]
                    if pred_y >= target_yield and (best_yield==0 or pred_y<best_yield):
                        best_yield = pred_y
                        best_combo = (N_opt, P_opt, K_opt)
        return best_combo, best_yield

    combo, achieved_yield = optimize_fertilizer(target_yield, soil_pH, rainfall, sunlight)

    if combo:
        st.success(f"🌱 {combo[0]} kg N, 🔵 {combo[1]} kg P, 🟤 {combo[2]} kg K\n📈 Yield: {achieved_yield:.2f} tons/ha")
    else:
        st.warning("No combination found to meet the target yield.")

    # Cost calculator
    st.subheader("💰 Fertilizer Cost Estimator")
    col_cost1, col_cost2, col_cost3 = st.columns(3)
    with col_cost1:
        cost_N = st.number_input("Cost of Nitrogen (₹/kg)", value=25)
    with col_cost2:
        cost_P = st.number_input("Cost of Phosphorus (₹/kg)", value=20)
    with col_cost3:
        cost_K = st.number_input("Cost of Potassium (₹/kg)", value=18)

    total_cost = combo[0]*cost_N + combo[1]*cost_P + combo[2]*cost_K
    st.metric("🧾 Estimated Fertilizer Cost", f"₹{total_cost:,.0f}")

    # Eco score
    safe_limits = {'Nitrogen':150, 'Phosphorus':70, 'Potassium':80}
    avg_ratio = np.mean([combo[0]/safe_limits['Nitrogen'], combo[1]/safe_limits['Phosphorus'], combo[2]/safe_limits['Potassium']])
    color = 'green' if avg_ratio <= 1 else 'orange' if avg_ratio <= 1.3 else 'red'
    message = "Eco-Friendly ✅" if avg_ratio <= 1 else "Caution ⚠️" if avg_ratio <= 1.3 else "Excessive ❌"

    fig, ax = plt.subplots(figsize=(4, 0.6))
    ax.barh([0], [avg_ratio], color=color, height=0.5)
    ax.set_xlim(0,2)
    ax.set_yticks([])
    ax.set_xlabel("Eco Usage Ratio")
    st.pyplot(fig)
    st.markdown(f"**{message}**")

# Visual Insights Tab
with tab3:
    st.subheader("📊 Feature Importance")
    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    fig_imp = px.bar(importance_df, x="Feature", y="Importance", color="Importance", color_continuous_scale="Viridis")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("📍 Fertilizer Composition")
    fig_pie = px.pie(names=['Nitrogen', 'Phosphorus', 'Potassium'], values=[combo[0], combo[1], combo[2]])
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("📈 Yield vs. Nitrogen")
    fig_scatter = px.scatter(data, x='Nitrogen', y='Yield', color='Phosphorus', size='Potassium')
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("🧪 Soil pH Distribution")
    fig_ph = px.histogram(data, x='Soil_pH', nbins=20)
    st.plotly_chart(fig_ph, use_container_width=True)

    st.subheader("🌦️ Climate Impact on Yield")
    fig_climate = px.scatter(data, x='Sunlight', y='Rainfall', size='Yield', color='Yield')
    st.plotly_chart(fig_climate, use_container_width=True)

# Download Report Tab
with tab4:
    st.subheader("📥 Export Results")
    report_df = pd.DataFrame({
        "Nitrogen":[combo[0]], "Phosphorus":[combo[1]], "Potassium":[combo[2]],
        "Soil_pH":[soil_pH], "Rainfall":[rainfall], "Sunlight":[sunlight],
        "Predicted_Yield":[achieved_yield]
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        report_df.to_excel(writer, index=False, sheet_name="Fertilizer_Recommendation")
    st.download_button("📄 Download Excel Report", data=buffer.getvalue(), file_name="fertilizer_report.xlsx")

# Feedback Tab
with tab5:
    st.subheader("📝 Feedback Form")
    name = st.text_input("Your Name")
    role = st.selectbox("Your Role", ["Farmer", "Agronomist", "Student", "Other"])
    comments = st.text_area("Your Feedback or Suggestions")

    if st.button("Submit Feedback"):
        st.success("✅ Thank you for your input! We’ll use it to improve the dashboard.")