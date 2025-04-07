import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from preprocessing import preprocess_data


st.set_page_config(
    page_title="방송매체 데이터 모델",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="auto",
)

# 저장된 모델 로드
model = joblib.load("best_xgb_model.pkl")


def main():
    st.title("설문 데이터 입력")

    # 기본 정보 입력
    # PID = st.number_input("PID", min_value=1, step=1)
    dm1 = st.selectbox("성별", ["남자", "여자"])
    DM2 = st.selectbox(
        "연령대", ["10대", "20대", "30대", "40대", "50대", "60대", "70세 이상"]
    )
    DM3 = st.selectbox(
        "지역",
        [
            "서울",
            "인천/경기",
            "부산/울산/경남",
            "광주/전라/제주",
            "대전/충청/세종",
            "대구/경북",
            "강원",
        ],
    )
    DM4 = st.selectbox(
        "직업", ["학생", "사무직", "서비스/판매직", "생산직", "기타", "주부", "무직"]
    )
    dm6 = st.selectbox(
        "학력", ["중/고등학생", "고졸이하", "대학생/대학원생", "대졸이상"]
    )
    dm7 = st.selectbox(
        "주거형태", ["단독주택", "아파트", "오피스텔", "기타", "다세대 및 연립주택"]
    )
    dm8 = st.selectbox(
        "가구 형태", ["독신가구", "1세대가구", "2세대가구", "3세대가구", "기타"]
    )

    Q416 = st.number_input(
        "스마트폰 일 평균 이용시간 (시간)",
        min_value=0,
        step=1,
        value=random.randint(0, 100),
    )
    Q417 = st.selectbox(
        "월평균 스마트폰 이용 요금",
        [
            "3만원 미만",
            "3만원 이상-4만원 미만",
            "4만원 이상-5만원 미만",
            "5만원 이상-6만원 미만",
            "6만원 이상-7만원 미만",
            "7만원 이상-10만원 미만",
            "10만원 이상",
        ],
        index=random.randint(0, 6),
    )

    # Q424A1 ~ Q424A4 선택지 업데이트
    importance_levels = [
        "매우 중요하다",
        "중요하다",
        "보통이다",
        "중요하지 않다",
        "전혀 중요하지 않다",
    ]
    Q424A1 = st.selectbox("Q424A1", importance_levels, index=random.randint(0, 4))
    Q424A2 = st.selectbox("Q424A2", importance_levels, index=random.randint(0, 4))
    Q424A3 = st.selectbox("Q424A3", importance_levels, index=random.randint(0, 4))
    Q424A4 = st.selectbox("Q424A4", importance_levels, index=random.randint(0, 4))

    # Q419A1 ~ Q419A8 선택지 업데이트
    frequency_levels = [
        "전혀안봄/이용안함",
        "일주일에 1-2일",
        "매일",
        "일주일에 3-4일",
        "한달에 1-3일",
        "일주일에 5-6일",
        "2-3달에 1-2일 이하",
    ]
    Q419A1 = st.selectbox("Q419A1", frequency_levels, index=random.randint(0, 6))
    Q419A2 = st.selectbox("Q419A2", frequency_levels, index=random.randint(0, 6))
    Q419A3 = st.selectbox("Q419A3", frequency_levels, index=random.randint(0, 6))
    Q419A4 = st.selectbox("Q419A4", frequency_levels, index=random.randint(0, 6))
    Q419A5 = st.selectbox("Q419A5", frequency_levels, index=random.randint(0, 6))
    Q419A6 = st.selectbox("Q419A6", frequency_levels, index=random.randint(0, 6))
    Q419A7 = st.selectbox("Q419A7", frequency_levels, index=random.randint(0, 6))
    Q419A8 = st.selectbox("Q419A8", frequency_levels, index=random.randint(0, 6))

    # Q263A1 ~ Q263A5 선택지 업데이트
    agreement_levels = [
        "보통이다",
        "그런 편이다",
        "그렇지 않은 편이다",
        "매우 그렇다",
        "전혀 그렇지 않다",
    ]
    Q263A1 = st.selectbox("Q263A1", agreement_levels, index=random.randint(0, 4))
    Q263A2 = st.selectbox("Q263A2", agreement_levels, index=random.randint(0, 4))
    Q263A3 = st.selectbox("Q263A3", agreement_levels, index=random.randint(0, 4))
    Q263A4 = st.selectbox("Q263A4", agreement_levels, index=random.randint(0, 4))
    Q263A5 = st.selectbox("Q263A5", agreement_levels, index=random.randint(0, 4))

    if st.button("예측 실행"):
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

        input_data = pd.DataFrame([input_dict])  # DataFrame 변환
        st.write("입력 데이터 (전처리 전):", input_data)
        # 예측 실행
        load_model = joblib.load("best_xgb_model.pkl")
        print("load_model 성공")
        print("====" * 20)
        processed_data = preprocess_data(input_data)
        st.write("입력 데이터 (전처리 후):", processed_data)
        print("pre_data 성공", processed_data)
        pred = load_model.predict(processed_data)[0]
        category_dic = {
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
        selected_categories = [
            category_dic[idx + 1] for idx, value in enumerate(pred) if value == 1
        ]

        # 출력
        st.write("선택된 카테고리:", selected_categories.values())


if __name__ == "__main__":
    main()
