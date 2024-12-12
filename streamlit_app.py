import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 장소검색 AI")
st.write(
    "신세계백화점 강남점에서 궁금한 것은 무엇이든 물어보세요."
    # "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
# openai_api_key = st.text_input("OpenAI API Key", type="password")
openai_api_key = st.secrets["OPENAI_API_KEY"]

from datetime import datetime
import pytz

import os
os.environ['OPENAI_API_KEY'] = openai_api_key

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    temperature=0,
    max_tokens=1024
)

PINECONE_KEY = 'pcsk_5LSZ4P_ANxevsPE5nS6idLg9CZPjMzPVNcj1JUHX6atoMxHq4zJjHZ6nBcW4PVbkgDZfa6'
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PINECONE_KEY)
pc_i = pc.Index("departments")

from sentence_transformers import SentenceTransformer
sro_embeddings = SentenceTransformer('jhgan/ko-sroberta-multitask')


shinsegae_gangnam = """ 
    매장명: 신세계백화점 강남점, 신강 혹은 신세계 강남 이라고도 부름
    전화번호: 1588-1234
    금요일,토요일,일요일은 연장영업. 영업시간의 경우 질문 요일을 참고하여 답변. 매장별 시간이 있는경우 매장 운영시간을 참고.
    정상 영업 10:30~20:00
    연장 영업 10:30~20:30
    휴점일: 2024년 기준 11월27일, 12월은 휴점일 없음
    지하1층 푸드코트는 백화점 영업시간과 동일함
    11층 전문 식당가영업 11:00~21:30 식당가 주문 마감 폐점 30분 전(단, 재료 소진 시 조기 마감 가능)
    [하우스 오브 신세계] 주중/주말 10:30~22:00 * 주문 마감 1시간 전(단, 재료 소진 시 조기 마감 가능)
    [B1 까사빠보/렌위치/브라우터/커피스니퍼/파이브가이즈] 10:30-21:00 (연장 영업 10:30-21:30)
    [신세계 팩토리스토어 강남점] 주중/주말 11:00~21:00
    [편의시설]
        수선실: 10층에 위치해 있으며, 남성복과 여성복 모두 수선 가능. 특히 명품 의류 수선 
        상품권 교환처: 8층 고객서비스센터에 위치. 이곳에서 상품권 교환이 가능하며, 키오스크를 통해 교환 절차를 진행할 수 있습니다. 
        수유실(리틀라운지): 10층에 위치해 있으며, 유아휴게실로서 수유 및 기저귀 교환 등의 편의시설을 제공합니다.
"""

def get_data_from_vector_db(query):
    query_embedding = sro_embeddings.encode(query).tolist()

    # Pinecone에서 유사한 항목 검색
    results = pc_i.query(
        vector=query_embedding,
        top_k=20,
        include_metadata=True,
        filter={
            # "area": "현대백화점 판교점"
            "area": "신세계백화점 강남점"
        }
    )

    store_text = ''
    # 검색 결과 출력
    for match in results['matches']:
        print(f"Score: {match['score']}, Name: {match['metadata']['name']}, Category: {match['metadata']['category']}, Floor: {match['metadata']['floor']}, Info: {match['metadata']['info']}")
        store_text += f"매장명: {match['metadata']['name']}, Category: {match['metadata']['category']}, Floor: {match['metadata']['floor']} Infomation: {match['metadata']['info']}\n"
    
    return store_text


# Create an OpenAI client.
# client = OpenAI(api_key=OPENAI_API_KEY)

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("무엇을 찾고있나요?"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response using the OpenAI API.
    # stream = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": m["role"], "content": m["content"]}
    #         for m in st.session_state.messages
    #     ],
    #     stream=True,
    # )

    store_text = get_data_from_vector_db(prompt)
    context = shinsegae_gangnam

    korea_timezone = pytz.timezone('Asia/Seoul')
    today = datetime.now(korea_timezone).strftime("%Y-%m-%d %H시 (%A)")
    # today = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
    location = "신세계 강남점"
    enhanced_prompt = f"""
        아래 정보를 참고해서 질문에 답변을 합니다. 지금 질문하는 사람은 해당 백화점 안에 있습니다.:
        오늘: {today}
        위치: {location}
        백화점 정보: {context}
        질문 관련 매장 정보: {store_text}
        질문: {prompt}
        답변:
    """
    stream = llm(enhanced_prompt).content

    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    with st.chat_message("assistant"):
        response = st.write(stream)
    # st.session_state.messages.append({"role": "assistant", "content": response})

    # Add assistant's response to session state
    st.session_state.messages.append({"role": "assistant", "content": stream})

# Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])
