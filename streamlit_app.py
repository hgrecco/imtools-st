import streamlit as st

import _tools

st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

st.write("# Welcome! 👋")

st.sidebar.success("Select an app above.")

st.markdown(
"""
I’ve put together this collection of tools to streamline and support various image analysis and visualization tasks. I hope you find them helpful in your own work or projects. If you have any questions, suggestions, or just want to connect, feel free to reach out — I’d love to hear from you!

Hernán Grecco

🔗 mail: [hgrecco@df.uba.ar](mailto:hgrecco@df.uba.ar)

🔗 twitter/X: [@GreccoHernan](https://twitter.com/GreccoHernan)

"""
)


with st.expander("Packages"):
    st.text("  \n".join([f"{k}: {v}" for k, v in _tools.versions()]))
        
