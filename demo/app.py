import streamlit as st

from blueprint.hello import hello

st.title("Blueprint Demo")

st.write(hello())
