import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from preprocessing import preprocess_data


st.set_page_config(
    page_title="맞춤 동영상 카테고리 추천",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="auto",
)

try:
    model = joblib.load("best_xgb_model.pkl")
except FileNotFoundError:
    st.error(
        "Error: Model file 'best_xgb_model.pkl' not found. Please ensure the model file is in the correct directory."
    )
    st.stop()  # Stop execution if model isn't found
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()


def local_css(file_name):
    try:
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback or default styles if CSS file is missing
        st.warning(f"'{file_name}' not found. Using default styles.")
        # Embed some basic default styles directly if needed
        st.markdown(
            """
        <style>
            /* Add some padding to the main block */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                padding-left: 3rem;
                padding-right: 3rem;
            }
            /* Style the button */
            .stButton>button {
                color: white;
                background-color: #FF4B4B; /* Streamlit Red */
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 0.25rem;
                font-size: 1rem;
                width: 100%; /* Make button wider */
                margin-top: 1rem;
            }
            .stButton>button:hover {
                background-color: #E03C3C;
                color: white;
            }
            /* Style expanders */
            .stExpander {
                border: 1px solid #e6e6e6;
                border-radius: 0.25rem;
                margin-bottom: 1rem;
            }
            .stExpander header {
                 font-weight: bold;
                 background-color: #f5f5f5; /* Light background for header */
                 padding: 0.5rem 1rem;
            }
             /* Center the results */
            .results-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 1.5rem;
                background-color: #f0f2f6;
                border-radius: 0.5rem;
                margin-top: 2rem;
            }
            .results-container .result-text {
                font-size: 1.2rem;
                font-weight: bold;
                color: #0e1117; /* Dark text color */
                text-align: center;
            }
            .results-container .categories {
                font-size: 1.1rem;
                color: #FF4B4B; /* Highlight color */
                text-align: center;
                margin-top: 0.5rem;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )


local_css("style.css")


GENDER_OPTIONS = ["남자", "여자"]
AGE_OPTIONS = ["10대", "20대", "30대", "40대", "50대", "60대", "70세 이상"]
REGION_OPTIONS = [
    "서울",
    "인천/경기",
    "부산/울산/경남",
    "광주/전라/제주",
    "대전/충청/세종",
    "대구/경북",
    "강원",
]
JOB_OPTIONS = ["학생", "사무직", "서비스/판매직", "생산직", "기타", "주부", "무직"]
EDU_OPTIONS = ["중/고등학생", "고졸이하", "대학생/대학원생", "대졸이상"]
HOUSING_OPTIONS = ["단독주택", "아파트", "오피스텔", "기타", "다세대 및 연립주택"]
FAMILY_OPTIONS = ["독신가구", "1세대가구", "2세대가구", "3세대가구", "기타"]
PHONE_COST_OPTIONS = [
    "3만원 미만",
    "3만원 이상-4만원 미만",
    "4만원 이상-5만원 미만",
    "5만원 이상-6만원 미만",
    "6만원 이상-7만원 미만",
    "7만원 이상-10만원 미만",
    "10만원 이상",
]
IMPORTANCE_LEVELS = [
    "전혀 중요하지 않다",
    "중요하지 않다",
    "보통이다",
    "중요하다",
    "매우 중요하다",
]
FREQUENCY_LEVELS = [
    "전혀안봄/이용안함",
    "2-3달에 1-2일 이하",
    "한달에 1-3일",
    "일주일에 1-2일",
    "일주일에 3-4일",
    "일주일에 5-6일",
    "매일",
]
AGREEMENT_LEVELS = [
    "전혀 그렇지 않다",
    "그렇지 않은 편이다",
    "보통이다",
    "그런 편이다",
    "매우 그렇다",
]
CATEGORY_DIC = {
    1: "스포츠",
    2: "웹드라마",
    3: "푸드",
    4: "웹예능",
    5: "음악/댄스",
    6: "시사/현장",
    7: "뷰티",
    8: "토크/캠방",
    9: "브이로그",
    10: "게임",
    11: "교육/학습",
    12: "종교",
    13: "영화",
    14: "기타",
}


# --- App Layout ---
st.title("🎬 맞춤 동영상 카테고리 추천 AI")
st.markdown(
    "간단한 설문을 통해 **가장 관심 있을 만한 동영상 카테고리**를 예측해 드립니다."
)
st.markdown("---")  # Visual separator

left_col, right_col = st.columns(2)

with left_col:
    st.header("👤 기본 정보")
    with st.expander("인구통계학적 정보", expanded=True):
        dm1 = st.selectbox("성별", GENDER_OPTIONS)
        DM2 = st.selectbox("연령대", AGE_OPTIONS)
        DM3 = st.selectbox("지역", REGION_OPTIONS)
        DM4 = st.selectbox("직업", JOB_OPTIONS)
        dm6 = st.selectbox("학력", EDU_OPTIONS)
        dm7 = st.selectbox("주거형태", HOUSING_OPTIONS)
        dm8 = st.selectbox("가구 형태", FAMILY_OPTIONS)

    st.header("📱 스마트폰 이용")
    with st.expander("스마트폰 이용 시간 및 요금"):
        Q416 = st.slider(
            "일 평균 스마트폰 이용 시간 (시간)", 0, 24, 12
        )  # Default value 12
        Q417 = st.selectbox(
            "월평균 스마트폰 이용 요금", PHONE_COST_OPTIONS, index=3
        )  # Example default index

    with st.expander("스마트폰 기능별 중요도"):
        Q424A1 = st.selectbox(
            "뉴스/정보 검색 및 이용", IMPORTANCE_LEVELS, index=random.randint(0, 4)
        )
        Q424A2 = st.selectbox(
            "동영상/음성 콘텐츠 시청/청취",
            IMPORTANCE_LEVELS,
            index=random.randint(0, 4),
        )
        Q424A3 = st.selectbox(
            "커뮤니케이션 (메신저, SNS 등)",
            IMPORTANCE_LEVELS,
            index=random.randint(0, 4),
        )
        Q424A4 = st.selectbox("쇼핑", IMPORTANCE_LEVELS, index=random.randint(0, 4))

with right_col:
    st.header("📺 시청 습관 및 선호도")
    with st.expander("스마트폰 콘텐츠 이용 빈도"):
        Q419A1 = st.selectbox(
            "TV프로그램 시청 빈도", FREQUENCY_LEVELS, index=random.randint(0, 6)
        )
        Q419A2 = st.selectbox(
            "라디오 청취 빈도", FREQUENCY_LEVELS, index=random.randint(0, 6)
        )
        Q419A3 = st.selectbox(
            "영화 보기 빈도", FREQUENCY_LEVELS, index=random.randint(0, 6)
        )
        Q419A4 = st.selectbox(
            "뉴스/정보 검색 빈도", FREQUENCY_LEVELS, index=random.randint(0, 6)
        )
        Q419A5 = st.selectbox(
            "음악 듣기 빈도", FREQUENCY_LEVELS, index=random.randint(0, 6)
        )
        Q419A6 = st.selectbox(
            "게임하기 빈도", FREQUENCY_LEVELS, index=random.randint(0, 6)
        )
        Q419A7 = st.selectbox(
            "e-book 읽기 빈도", FREQUENCY_LEVELS, index=random.randint(0, 6)
        )
        Q419A8 = st.selectbox(
            "동영상 시청 빈도 (TV/영화 제외)",
            FREQUENCY_LEVELS,
            index=random.randint(0, 6),
        )

    with st.expander("TV 시청 관련 태도"):
        Q263A1 = st.selectbox(
            "혼자 TV 보기를 좋아한다", AGREEMENT_LEVELS, index=random.randint(0, 4)
        )
        Q263A2 = st.selectbox(
            "몰아서 보는 것을 좋아한다", AGREEMENT_LEVELS, index=random.randint(0, 4)
        )
        Q263A3 = st.selectbox(
            "본방사수를 좋아한다", AGREEMENT_LEVELS, index=random.randint(0, 4)
        )
        Q263A4 = st.selectbox(
            "습관적으로 TV를 본다", AGREEMENT_LEVELS, index=random.randint(0, 4)
        )
        Q263A5 = st.selectbox(
            "TV 보면서 소통하는 것을 좋아한다",
            AGREEMENT_LEVELS,
            index=random.randint(0, 4),
        )


st.write("")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button("🚀 추천 카테고리 예측하기!")

if predict_button:
    input_dict = {
        "dm1": dm1,
        "DM2": DM2,
        "DM3": DM3,
        "DM4": DM4,
        "dm6": dm6,
        "dm7": dm7,
        "dm8": dm8,
        "Q416": Q416,
        "Q417": Q417,
        "Q424A1": Q424A1,
        "Q424A2": Q424A2,
        "Q424A3": Q424A3,
        "Q424A4": Q424A4,
        "Q419A1": Q419A1,
        "Q419A2": Q419A2,
        "Q419A3": Q419A3,
        "Q419A4": Q419A4,
        "Q419A5": Q419A5,
        "Q419A6": Q419A6,
        "Q419A7": Q419A7,
        "Q419A8": Q419A8,
        "Q263A1": Q263A1,
        "Q263A2": Q263A2,
        "Q263A3": Q263A3,
        "Q263A4": Q263A4,
        "Q263A5": Q263A5,
    }

    input_data = pd.DataFrame([input_dict])

    # --- Run Prediction ---
    try:
        with st.spinner("🧠 AI가 분석 중입니다... 잠시만 기다려주세요!"):

            processed_data = preprocess_data(input_data.copy())

            prediction_proba = model.predict_proba(processed_data)
            pred_binary = model.predict(processed_data)[0]

            selected_categories = [
                CATEGORY_DIC[idx + 1]
                for idx, value in enumerate(pred_binary)
                if value == 1
            ]

        # --- Display Results ---
        if selected_categories:
            # 결과 제목 부분 - 아이콘 추가 및 텍스트 수정
            # st.markdown(
            #     '<div class="results-container">',
            #     unsafe_allow_html=True,  # 컨테이너 시작을 여기로 이동
            # )
            st.markdown(
                '<p class="result-text">✨ 당신을 위한 맞춤 추천 카테고리 ✨</p>',  # 아이콘 및 텍스트 변경
                unsafe_allow_html=True,
            )

            # 카테고리들을 개별 태그로 만들기
            tags_html = (
                '<div class="category-tags-container">'  # 태그들을 감싸는 컨테이너
            )
            for category in selected_categories:
                # 각 카테고리에 대한 태그 HTML 생성
                tags_html += f'<span class="category-tag">{category}</span>'
            tags_html += "</div>"  # 태그 컨테이너 닫기

            # 생성된 태그 HTML을 markdown으로 렌더링
            st.markdown(tags_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)  # results-container 닫기

            # (선택 사항) 예측 확률 등 추가 정보 표시 가능
            # st.write("Prediction Probabilities (example):", prediction_proba[0])

        else:
            # 추천 카테고리가 없을 경우 - 컨테이너 안에서 표시
            st.markdown(
                '<div class="results-container">', unsafe_allow_html=True
            )  # 컨테이너 시작
            st.warning("🤔 추천할 만한 특별한 카테고리를 찾지 못했습니다.")
            st.markdown("</div>", unsafe_allow_html=True)  # 컨테이너 닫기
        st.markdown("</div>", unsafe_allow_html=True)

    except AttributeError as e:
        st.error(
            f"Model Prediction Error: The model might not have a 'predict' or 'predict_proba' method, or the input data format is incorrect after preprocessing. Details: {e}"
        )
        st.info(
            "💡 Check that your `preprocess_data` function output matches the features the model was trained on (order and encoding)."
        )

    except ValueError as e:
        st.error(
            f"Data Format Error: The number of features in the preprocessed data might not match the model's expectations. Details: {e}"
        )
        st.info(
            "💡 Check that your `preprocess_data` function output matches the features the model was trained on (order and encoding)."
        )
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
st.markdown("---")
st.caption("Powered by Streamlit & XGBoost | Design Assistance by AI")
