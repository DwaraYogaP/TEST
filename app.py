import streamlit as st
import pandas as pd
import joblib,os


from sklearn.feature_extraction.text import TfidfVectorizer
data_train = pd.read_excel("/Book2.xlsx")
data_test = pd.read_excel("/Book1.xlsx")

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(data_train['Gejala'])
test_vectors = vectorizer.transform(data_test['Gejala'])

st.title("PREDIKSI DIAGNOSA PENYAKIT GIGI")

def load(file):
    load = joblib.load(open(os.path.join(file), "rb"))
    return load


model  = load("/gigi_clf.pkl")

teks = st.text_input("")

if st.button("Proses"):
            st.info("Results")
            teks_vector = vectorizer.transform([teks])
            hasil = model.predict(teks_vector)

            st.info(hasil[0])
