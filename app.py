
import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="AI Diabetes Risk Prediction",page_icon="🩺",layout="wide")

st.markdown("""
<style>
.title{font-size:38px;font-weight:bold;color:#1565C0;}
.footer{text-align:center;color:gray;}
</style>
""",unsafe_allow_html=True)

@st.cache_resource
def load():
    return pickle.load(open("diabetes_model.pkl","rb")), pickle.load(open("scaler.pkl","rb"))
model,scaler=load()

st.sidebar.title("🏥 Project")
st.sidebar.write("AI-Based Diabetes Risk Prediction")

st.markdown('<p class="title">🩺 AI-Based Diabetes Risk Prediction & Clinical Decision Support System</p>',unsafe_allow_html=True)
patient=st.text_input("Patient Name")
c1,c2=st.columns(2)
with c1:
    pregnancies=st.number_input("Pregnancies",0,20,1)
    glucose=st.number_input("Glucose",0,250,120)
    bp=st.number_input("Blood Pressure",0,150,70)
    skin=st.number_input("Skin Thickness",0,100,20)
with c2:
    insulin=st.number_input("Insulin",0,900,80)
    bmi=st.number_input("BMI",0.0,70.0,25.0)
    dpf=st.number_input("Diabetes Pedigree Function",0.0,3.0,0.5)
    age=st.number_input("Age",1,120,33)

if st.button("🔍 Predict Diabetes Risk"):
    x=np.array([[pregnancies,glucose,bp,skin,insulin,bmi,dpf,age]])
    xs=scaler.transform(x)
    pred=model.predict(xs)[0]
    try:
        prob=model.predict_proba(xs)[0][1]
    except:
        prob=float(pred)
    if pred:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
    st.metric("Risk Probability",f"{prob*100:.2f}%")
    st.progress(float(prob))
    st.subheader("Recommendations")
    if pred:
        st.write("- Consult a doctor\n- Exercise regularly\n- Maintain healthy diet")
    else:
        st.write("- Continue healthy lifestyle\n- Annual check-up")
    st.write(f"Patient: {patient or 'N/A'} | Age: {age} | BMI: {bmi}")
st.markdown("---")
st.markdown('<div class="footer">Developed using Streamlit</div>',unsafe_allow_html=True)
