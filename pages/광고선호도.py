import streamlit as st
import pandas as pd
import joblib
from preprocessing_sungwoo import load_and_preprocess_team2_data

# --- 페이지 설정 ---
st.set_page_config(
    page_title="사용자 정보 입력 설문",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="auto",
)

try:
    model = joblib.load("random_forest_model_sungwoo.pkl")
except FileNotFoundError:
    st.error(
        "Error: Model file 'random_forest_model_sungwoo.pkl' not found. Please ensure the model file is in the correct directory."
    )
    st.stop()  # Stop execution if model isn't found
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# --- 설문 제목 ---
st.title("📝 사용자 정보 입력")
st.write("모델 예측을 위해 아래 설문에 응답해주세요.")
st.markdown("---")

# --- 입력 옵션 정의 ---
gender_options = ["남자", "여자"]
age_options = ["10대", "20대", "30대", "40대", "50대", "60대", "70세 이상"]
region_options = [
    "서울",
    "인천/경기",
    "부산/울산/경남",
    "광주/전라/제주",
    "대전/충청/세종",
    "대구/경북",
    "강원",
]
job_options = ["학생", "사무직", "서비스/판매직", "생산직", "기타", "주부", "무직"]
income_options = [
    "100만원 미만",
    "100-199만원",
    "200-299만원",
    "300-399만원",
    "400만원 이상",
]
edu_options = ["중/고등학생", "고졸이하", "대학생/대학원생", "대졸이상"]
housing_options = ["독신가구", "1세대가구", "2세대가구", "3세대가구", "기타"]
city_type_options = ["대도시", "중소도시", "군지역"]
q531a9_options = ["예", "아니오", "모름/무응답"]

# Q353M: 선택 가능한 이유 목록 (체크박스용, '해당 없음' 제외)
selectable_q353_reasons = [
    "방송 못 본 것을 보기 위해서",
    "방송 본 것을 다시 보고 싶어서",
    "이동 시 시간을 활용하기 위해서",
    "광고를 보기 싫어서",  # 중요 옵션
    "원하는 시간에 보기 위해서",
    "TV 수상기에 없는 장소에서 보기 위해서",
    "기타",
]
# Q353M 컬럼에 채울 기본값
default_q353_value = "무응답"

# --- 입력 폼 생성 ---
with st.form("survey_form"):
    st.header("👤 기본 인적 사항")
    col1, col2, col3 = st.columns(3)
    with col1:
        dm1_input = st.radio(
            "1. 성별 (dm1)", gender_options, key="dm1", horizontal=True
        )
        dm4_input = st.selectbox("4. 직업 (DM4)", job_options, key="dm4")
        dm8_input = st.selectbox("7. 가구 형태 (dm8)", housing_options, key="dm8")
    with col2:
        DM2_input = st.selectbox("2. 연령 (DM2)", age_options, key="DM2")
        DM5_input = st.selectbox(
            "5. 가구 소득 (월 평균) (DM5)", income_options, key="DM5"
        )
        DM11_input = st.selectbox(
            "8. 거주 지역 유형 (DM11)", city_type_options, key="DM11"
        )
    with col3:
        DM3_input = st.selectbox("3. 거주 지역 (DM3)", region_options, key="DM3")
        dm6_input = st.selectbox("6. 학력 (dm6)", edu_options, key="dm6")

    st.markdown("---")
    st.header("📺 미디어 이용 행태")

    # --- Q353M 입력 방식 변경 (Checkbox) ---
    st.markdown(
        "##### 9. 미디어/OTT 서비스 이용 중단 또는 변경 이유 (해당하는 것을 모두 선택)"
    )
    # 체크박스 상태 저장을 위한 딕셔너리
    q353_checkbox_states = {}
    # 체크박스를 보기 좋게 여러 열로 나눔 (예: 3열)
    reason_cols = st.columns(3)
    col_idx = 0
    for reason in selectable_q353_reasons:
        # 각 이유에 대한 체크박스 생성, key는 고유해야 함
        # 현재 상태를 딕셔너리에 저장
        with reason_cols[col_idx % 3]:  # 3개의 컬럼에 순환 배치
            q353_checkbox_states[reason] = st.checkbox(reason, key=f"q353_cb_{reason}")
        col_idx += 1
    # ----------------------------------------

    st.markdown("##### 10. 광고 기반 무료 서비스 이용 의향 (Q531A9)")
    st.caption("예: 광고를 보는 대신 무료로 콘텐츠를 이용하시겠습니까?")
    Q531A9_input = st.radio(
        "", q531a9_options, key="Q531A9", horizontal=True, label_visibility="collapsed"
    )

    st.markdown("---")
    submitted = st.form_submit_button("📝 설문 제출")

# --- 제출 후 처리 ---
if submitted:
    # 1. 기본 인적사항 값 가져오기
    survey_data = {
        "dm1": dm1_input,
        "DM2": DM2_input,
        "DM3": DM3_input,
        "DM4": dm4_input,
        "DM5": DM5_input,
        "dm6": dm6_input,
        "dm8": dm8_input,
        "DM11": DM11_input,
        "Q531A9": Q531A9_input,
    }

    # --- Q353M 체크박스 결과 처리 ---
    # 선택된 이유들만 리스트로 추출
    selected_reasons = [
        reason for reason, checked in q353_checkbox_states.items() if checked
    ]

    # Q353M_1 ~ Q353M_6 컬럼 값 생성
    q_cols_outputs = {}
    num_selected = len(selected_reasons)
    for i in range(1, 7):  # 1부터 6까지
        col_name = f"Q353M_{i}"
        if i <= num_selected:
            # 선택된 이유가 있으면 순서대로 할당
            q_cols_outputs[col_name] = selected_reasons[i - 1]
        else:
            # 선택된 이유 개수를 넘어서면 기본값 할당
            q_cols_outputs[col_name] = default_q353_value
    # --------------------------------

    # 기본 데이터와 Q353M 결과 합치기
    survey_data.update(q_cols_outputs)

    # 2. Pandas DataFrame으로 변환
    input_df = pd.DataFrame([survey_data])

    # 3. 입력 결과 확인용 출력
    st.subheader("✍️ 제출된 설문 내용")
    # 컬럼 순서를 보기 좋게 재정렬하여 표시
    display_order = [
        "dm1",
        "DM2",
        "DM3",
        "DM4",
        "DM5",
        "dm6",
        "dm8",
        "DM11",
        "Q353M_1",
        "Q353M_2",
        "Q353M_3",
        "Q353M_4",
        "Q353M_5",
        "Q353M_6",
        "Q531A9",
    ]
    st.dataframe(input_df[display_order])

    st.success("설문이 성공적으로 제출되었습니다!")

    # 여기에 추가적으로 input_df를 사용한 전처리 함수 호출 또는 모델 예측 코드 삽입 가능
    processed_df = load_and_preprocess_team2_data(
        input_df
    )  # 이전 단계에서 만든 전처리 함수
    if processed_df is not None:
        st.subheader("⚙️ 전처리 결과")
        st.dataframe(processed_df)
# --- 제출 후 처리 ---
if submitted:
    # ... (기존 코드: survey_data 생성, input_df 생성 및 표시) ...

    st.success("설문이 성공적으로 제출되었습니다!")

    st.markdown("---")  # 구분선 추가
    st.subheader("⚙️ 데이터 전처리 수행")
    st.write(
        "입력된 설문 내용을 기반으로 모델이 이해할 수 있는 형태로 데이터를 변환합니다."
    )

    # 전처리 함수 호출 (파일 경로 대신 DataFrame 전달하도록 수정 필요)
    # *** 중요: load_and_preprocess_team2_data 함수가 DataFrame을 받도록 수정되었거나,
    #     파일 경로 대신 DataFrame을 처리하는 별도의 함수가 있다고 가정합니다.
    #     만약 원래 함수가 파일 경로만 받는다면, 해당 함수를 수정하거나
    #     이 Streamlit 앱 내에서 동일한 전처리 로직을 직접 수행해야 합니다.
    #     여기서는 input_df를 직접 처리하는 함수라고 가정합니다.
    try:
        # preprocess_input_dataframe 함수를 사용한다고 가정 (이전 답변에서 생성)
        # 만약 preprocessing_sungwoo.py 에 DataFrame 처리 함수가 있다면 그것 사용
        # 예시: processed_df = preprocess_input_dataframe(input_df)
        processed_df = load_and_preprocess_team2_data(
            input_df
        )  # 함수 이름은 유지, DataFrame 입력 가능 가정

        if processed_df is not None:
            st.write("✅ 전처리 완료!")
            st.dataframe(processed_df)
            st.info(
                """
                ⚠️ **참고:**
                1.  이 전처리 결과는 입력된 단일 데이터 포인트에 대해서만 수행되었습니다.
                2.  실제 모델 예측을 위해 원-핫 인코딩 등의 결과로 생성된 **컬럼들이 모델 학습 시의 컬럼과 정확히 일치**해야 합니다.
                """
            )

            # --- 모델 예측 수행 ---
            st.markdown("---")  # 구분선 추가
            st.subheader("🔮 모델 예측 결과")

            try:
                # 모델 예측 (processed_df가 모델 입력 형식과 일치해야 함)
                prediction = model.predict(processed_df)
                prediction_proba = model.predict_proba(processed_df)  # 확률값 예측

                # 예측 결과 해석 (모델이 예측하는 값에 따라 수정 필요)
                # 예: 0: 비선호, 1: 보통, 2: 선호 라고 가정 (OrdinalEncoder 순서 기반)
                prediction_map = {0: "비선호", 1: "보통", 2: "선호"}
                predicted_label = prediction_map.get(
                    prediction[0], "알 수 없음"
                )  # 첫번째 예측 결과 사용

                # 예측 확률 표시 (클래스 순서에 맞게)
                # model.classes_ 를 확인하여 순서를 아는 것이 가장 좋음
                # 여기서는 0, 1, 2 순서라고 가정
                prob_dislike = prediction_proba[0][0] * 100
                prob_neutral = prediction_proba[0][1] * 100
                prob_like = prediction_proba[0][2] * 100

                # 결과 시각화 (st.metric 또는 st.write 등 활용)
                st.metric(label="예측된 광고 선호도", value=predicted_label)

                st.write("##### 예측 확률:")
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                with prob_col1:
                    st.metric(label="비선호 확률", value=f"{prob_dislike:.1f}%")
                with prob_col2:
                    st.metric(label="보통 확률", value=f"{prob_neutral:.1f}%")
                with prob_col3:
                    st.metric(label="선호 확률", value=f"{prob_like:.1f}%")

                # 추가 설명
                if predicted_label == "선호":
                    st.success(
                        "이 사용자는 광고 기반 서비스에 대해 긍정적인 반응을 보일 가능성이 높습니다."
                    )
                elif predicted_label == "보통":
                    st.info(
                        "이 사용자는 광고 기반 서비스에 대해 중립적인 반응을 보일 수 있습니다."
                    )
                else:  # 비선호
                    st.warning(
                        "이 사용자는 광고 기반 서비스에 대해 부정적인 반응을 보일 가능성이 높습니다."
                    )

            except AttributeError as e:
                st.error(
                    f"모델 예측 오류: 모델 객체에 'predict' 또는 'predict_proba' 메서드가 없거나 잘못되었습니다. 모델 파일을 확인하세요. ({e})"
                )
            except ValueError as e:
                st.error(
                    f"모델 입력 오류: 전처리된 데이터의 컬럼 수나 형식이 모델이 학습된 데이터와 다릅니다. 전처리 함수 또는 모델을 확인하세요. ({e})"
                )
                st.dataframe(processed_df)  # 어떤 데이터가 입력되었는지 보여주기
                st.write(
                    "모델 예상 컬럼 수:",
                    (
                        model.n_features_in_
                        if hasattr(model, "n_features_in_")
                        else "알 수 없음"
                    ),
                )  # 모델이 기대하는 특성 수 표시
            except Exception as e:
                st.error(f"예측 중 예상치 못한 오류 발생: {e}")

        else:
            st.error("데이터 전처리 중 오류가 발생하여 모델 예측을 수행할 수 없습니다.")

    except ImportError:
        st.error(
            "오류: 'preprocessing_sungwoo' 모듈 또는 'load_and_preprocess_team2_data' 함수를 찾을 수 없습니다. 파일 이름과 함수 이름을 확인하세요."
        )
    except Exception as e:
        st.error(f"전처리 함수 실행 중 오류 발생: {e}")
        st.info("전처리 함수의 로직이나 입력 데이터 형식을 확인해보세요.")
