import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response

# 제목 설정
st.set_page_config(page_title="소득세 챗봇", page_icon=":shark:")

st.title("🦈 소득세 챗봇")
st.caption("소득세 관련 질문에 대해 답변해드립니다!")

load_dotenv()

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 메시지 출력
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    # 사용자 질문 출력
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    # AI 답변 생성
    with st.spinner("답변을 생성중입니다"):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})

# session state = 전역변수
