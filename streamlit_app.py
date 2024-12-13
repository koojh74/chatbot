import streamlit as st
from openai import OpenAI


def set_title_style():
    # CSS를 사용하여 타이틀 영역 스타일링
    st.markdown("""
        <style>
        .title-container {
            background-color: #E3F2FD;  /* 연한 하늘색 */
            padding: 0.2rem;
            border-radius: 6px;
            margin-bottom: 0.1rem;
            text-align: center;
        }
        .title-text {
            color: #1976D2;  /* 진한 파란색 글씨 */
            font-size: 1.8rem;
            font-weight: bold;
            margin: 0.1;
        }
        /* 전체 페이지 여백 조정 */
        .block-container {
            padding-top: 3rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0.1rem !important;
            padding-right: 0.1rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

def custom_title(title_text):
    st.markdown(f"""
        <div class="title-container">
            <h2 class="title-text">{title_text}</h2>
        </div>
        """, unsafe_allow_html=True)


set_title_style()
custom_title("MAP Assistant")

# Show title and description.
# st.title("MAP Assistant")
st.write(
    "아래 장소에서 궁금한건 뭐든지 물어보세요. (예: 식당 영업시간, 등산화, 피자가 먹고싶어, 상품권 등)"
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

hyundai_pangyo = """
    매장명: 현대백화점 판교점, 혹은 판교 현대라고도 부름
    대표전화: 031-5170-2233
    영업시간:
        평일(월~목): 10:30 ~ 20:00
        주말(금~일): 10:30 ~ 20:30
        식당가: 10:30 ~ 22:00
    휴점일: 2024년 12월은 휴점일 없음
    편의시설:(장소명,층,전화번호,세부내역)
        물품보관,지하1층,031-5170-1000,짐이나 쇼핑한 물건을 보관할 수 있는 무인 락커. 냉장 상품 보관
        스마트서비스박스(무인물품보관함),7층,031-5170-3302,쇼핑에 번거로운 짐 보관 번거로운 세탁 수선 접수와 택배수령
        배송접수,지하1층,031-5170-1017,식품관 구매 물품을 대상으로 근거리와 원거리 배송 서비스를 제공
        교환·환불,지하1층,031-5170-2088,식품관에서 구매하신 물품의 교환 및 환불 업무를 진행
        유모차데스크,지하1층,031-5170-3290,36개월 이하의 유아를 동반한 고객님들께 청결하게 관리된 유모차를 대여
        유모차데스크,5층,031-5170-3290,36개월 이하의 유아를 동반한 고객님들께 청결하게 관리된 유모차를 대여
        사은데스크,10층,031-5170-3272,현대백화점 상품권 및 사은품 행사에 참여
        안내데스크,지하1층,031-5170-1114,층별 매장 / 쇼핑 안내 및 외국인 안내 서비스를 제공. 휴대용 스마트폰 충전기 대여
        문화센터,9층,031-5170-4560,문화센터 강좌 및 문화공연 접수 서비스를 제공
        의류수선실,4층 6층,4층(031-5170-2421~3) 6층(031-5170-2600~1),의류 수선실은 원하시는 상품으로 리폼, 수선, 재단 등을 실시
        시계/수리 및 클리닝,2층,031-5170-2242,시계 수리 및 클리닝 서비스를 제공. 해시계(시계 수선 : 031-5170-2242), 쿨화이트 (프리미엄 세탁 서비스 : 031-5170-2243)
        구두/피혁 수선,3층,031-5170-2241,레더플레인(구두/가방 等 피혁 상품 수선)
        유아휴게실,5층,031-5170-2700,유아를 동반한 고객님들께 쇼핑 중 쾌적하고 쉼터와 같은 편안한 휴식 공간. 라운지 내 수유실과 청결실, 수면실을 별도로 조성
        클럽데스크,10층,031-5170-3280,현대백화점 멤버십 CLUB WEDDING, 외국인 마케팅 (K-Card) 업무. 멤버십 가입 및 안내, 마일리지 적립, 고객 대상 이벤트/프로모션이 진행.
        회원서비스센터,10층,N-1588-4560,현대백화점 카드 신규발급 및 재발급, 분실신구, 대금납부 등 현대백화점 카드와 관련된 모든 서비스를 제공
        상품권 데스크,10F,031-5170-3100,현대백화점 상품권을 구매하시거나, 포인트 및 모바일 상품권을 지류 상품권으로 교환
        여행사 데스크,10층,031-5170-3521,여행과 관련한 궁금한 사항 상담해드립니다. 패키지/허니문/골프/단체여행/연수/가족여행/항공 및 호텔 등 예약
        고객상담실,10층,031-5170-4500,고객 상담을 위한 공간으로 백화점 서비스에 대한 불편사항을 친절하게 도와드림
        CAFE H,10층,음료 및 간식을 서비스하는 공간입니다. 현대백화점 APP 가입 고객께서는 무료 음료를 선택하여 드실 수 있
        커뮤니티 라운지,10층,,여행, 봉사, 미술, 뷰티, 영화 등 다양한 취미를 가진 현대백화점 동호회 회원을 위한 임직원과의 커뮤니티 공간
        토파즈홀(문화홀),10층,031-5170-3260,350석 규모의 고품격 문화 예술 공간으로 뮤지컬 / 연극 等 다양한 공연 및 콘서트를 위한 시설
        ATM,지하1층+5층+10층,,지하 1층-신한은행 5층-우리은행 10층-국민은행·하나은행 ATM이 설치
        통합수선실,10층,031-5170-3284,판교점에서 구매하신 상품 중 미입점/퇴점 등의 사유로 영업을 하고 있지 않은 브랜드 상품의 수선/교환/환불 등 사후 서비스
"""

def get_data_from_vector_db(query, area):
    query_embedding = sro_embeddings.encode(query).tolist()

    # Pinecone에서 유사한 항목 검색
    results = pc_i.query(
        vector=query_embedding,
        top_k=20,
        include_metadata=True,
        filter={
            # "area": "현대백화점 판교점"
            # "area": "신세계백화점 강남점"
            "area": area
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

if "area" not in st.session_state:
    st.session_state.area = "신세계백화점 강남점"

area = st.selectbox(
    "백화점을 선택하세요",
    options=["신세계백화점 강남점", "현대백화점 판교점"],
    key="option_select"
)
st.session_state.area = area

# st.divider()

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

    store_text = get_data_from_vector_db(prompt, st.session_state.area)

    if st.session_state.area == '현대백화점 판교점':
        context = hyundai_pangyo
    else:
        context = shinsegae_gangnam
    

    korea_timezone = pytz.timezone('Asia/Seoul')
    today = datetime.now(korea_timezone).strftime("%Y-%m-%d %H시 (%A)")
    # today = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
    location = st.session_state.area
    enhanced_prompt = f"""
        아래 정보를 참고해서 질문에 답변을 합니다. 지금 질문하는 사람은 해당 백화점 안에 있습니다.
        매장내 판매 품목과 관련하여 과도한 추정은 하지 않습니다.
        매장 정보를 알려줄때는 층 정보를 꼭 함께 알려줍니다.:
        오늘: {today}
        위치: {location}
        백화점 정보: {context}
        질문 관련 매장 정보: {store_text}
        질문: {prompt}
        답변:
    """
    stream = f'[{location}] ' + llm(enhanced_prompt).content

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
