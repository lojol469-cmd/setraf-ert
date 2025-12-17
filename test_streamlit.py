import streamlit as st
import pandas as pd
import numpy as np

st.title("Test SETRAF")
st.write("Si vous voyez ce message, Streamlit fonctionne!")

# Test basique
st.header("Test des imports")
try:
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    st.write("✅ Pandas OK")
    st.dataframe(df)
except Exception as e:
    st.error(f"❌ Pandas error: {e}")

try:
    arr = np.array([1, 2, 3])
    st.write("✅ NumPy OK")
except Exception as e:
    st.error(f"❌ NumPy error: {e}")

st.write("Test terminé")
