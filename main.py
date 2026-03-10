import streamlit as st
from graph.graph import app 

st.set_page_config(page_title="Akin Capar AI", page_icon="🤖", layout="centered")

st.title("🤖 Akin Çapar - AI Asistant")
st.markdown("Hi! I'm Akin's digital assistant. I'm here to help with your resume, experience, or any general questions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Which university did Akin graduate from?"):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking... (Documents are being scanned)"):
            try:
                inputs = {"question": prompt}
                
                result = app.invoke(inputs)
                
                answer = result.get("generation", "Sorry, try another question.")
                
                st.markdown(answer)
                
                # Cevabı hafızaya ekle
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"System error: {e}")