import streamlit as st
import pandas as pd
import joblib
import random  # random 모듈 추가

# from preprocessing_sungwoo import load_and_preprocess_team2_data # 이 import는 유지
# --- preprocessing_sungwoo.py 함수 정의 (이전 답변 내용 붙여넣기) ---
from sklearn.preprocessing import OrdinalEncoder  # 전처리 함수 내 필요


def load_and_preprocess_team2_data(input_df):
    # (이전 답변에서 제공된 수정된 전처리 함수 전체 내용 복사/붙여넣기)
    """
    Streamlit에서 입력받은 단일 행 DataFrame을 전처리하여
    지정된 27개 feature 컬럼 구조로 반환합니다. ... (함수 내용 전체) ...
    """
    # 0. 필요한 원본 컬럼 정의 (입력 df에 있는지 확인용)
    save_cols = [
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
    missing_cols = [col for col in save_cols if col not in input_df.columns]
    if missing_cols:
        st.error(
            f"입력 데이터에 필요한 컬럼이 누락되었습니다: {', '.join(missing_cols)}"
        )
        return None

    df_processed = input_df[save_cols].copy()

    # 1. 컬럼명 변경
    rename_map = {
        "dm1": "성별",
        "DM2": "나이대",
        "DM3": "지역",
        "DM4": "직업",
        "DM5": "소득",
        "dm6": "학력",
        "dm8": "가구형태",
        "DM11": "도시유형",
    }
    df_processed.rename(columns=rename_map, inplace=True)

    # 2. '광고비선호' 상태 확인
    is_ad_averse = False
    q353_cols_original = [f"Q353M_{i}" for i in range(1, 7)]
    for col in q353_cols_original:
        if df_processed.loc[0, col] == "광고를 보기 싫어서":
            is_ad_averse = True
            break

    # 3. '광고선호' 상태 결정
    q531a9_val = df_processed.loc[0, "Q531A9"]
    if q531a9_val in ["아니오", "모름/무응답"]:
        initial_ad_preference = "보통"
    elif q531a9_val == "예":
        initial_ad_preference = "선호"
    else:
        initial_ad_preference = "보통"

    # 4. 논리 충돌 확인
    if is_ad_averse and initial_ad_preference == "선호":
        st.error(
            "입력 오류: '광고 보기 싫음'과 '광고 기반 서비스 이용 의향 있음'은 동시에 선택될 수 없습니다."
        )
        return None

    # 5. 최종 '광고선호' 결정
    if is_ad_averse and initial_ad_preference == "보통":
        final_ad_preference = "비선호"
    else:
        final_ad_preference = initial_ad_preference
    df_processed["광고선호"] = final_ad_preference

    # 6. 소득 '무응답' 확인
    if df_processed.loc[0, "소득"] == "무응답":
        st.error("입력 오류: '소득'은 '무응답'일 수 없습니다.")
        return None

    # 7. 순서형 인코딩
    ordinal_map = {
        "나이대": ["10대", "20대", "30대", "40대", "50대", "60대", "70세 이상"],
        "소득": [
            "100만원 미만",
            "100-199만원",
            "200-299만원",
            "300-399만원",
            "400만원 이상",
        ],
        "학력": ["중/고등학생", "고졸이하", "대학생/대학원생", "대졸이상"],
        "광고선호": ["비선호", "보통", "선호"],
    }
    ordinal_encoded_cols = []
    for col, categories in ordinal_map.items():
        if col in df_processed.columns:
            try:
                encoder = OrdinalEncoder(categories=[categories])
                col_value = df_processed[[col]]
                encoded_col_name = f"{col}_encoded"
                df_processed[encoded_col_name] = encoder.fit_transform(
                    col_value
                ).astype("int64")
                ordinal_encoded_cols.append(encoded_col_name)
            except ValueError as e:
                st.error(
                    f"'{col}' 컬럼 순서형 인코딩 오류: 입력값 '{df_processed.loc[0, col]}' 확인 필요. ({e})"
                )
                return None
        else:
            st.warning(
                f"Warning: 순서형 인코딩 대상 컬럼 '{col}'이(가) 데이터에 없습니다."
            )

    # 8. 원핫 인코딩
    onehot_cols = ["성별", "지역", "직업", "가구형태", "도시유형"]
    df_processed = pd.get_dummies(
        df_processed, columns=onehot_cols, drop_first=False, dtype=bool
    )

    # 9. 최종 컬럼 선택 및 생성/정리
    target_feature_columns = [
        "나이대_encoded",
        "소득_encoded",
        "학력_encoded",
        "성별_남자",
        "성별_여자",
        "지역_강원",
        "지역_광주/전라/제주",
        "지역_대구/경북",
        "지역_대전/충청/세종",
        "지역_부산/울산/경남",
        "지역_서울",
        "지역_인천/경기",
        "직업_기타",
        "직업_무직",
        "직업_사무직",
        "직업_생산직",
        "직업_서비스/판매직",
        "직업_주부",
        "직업_학생",
        "가구형태_1세대가구",
        "가구형태_2세대가구",
        "가구형태_3세대가구",
        "가구형태_기타",
        "가구형태_독신가구",
        "도시유형_군지역",
        "도시유형_대도시",
        "도시유형_중소도시",
    ]

    final_df = pd.DataFrame()
    for col in target_feature_columns:
        if col in df_processed.columns:
            final_df[col] = df_processed[col]
        elif col.endswith("_encoded"):
            st.error(f"오류: 필요한 순서형 인코딩 컬럼 '{col}'이 생성되지 않았습니다.")
            return None
        else:
            final_df[col] = False

    final_df = final_df[target_feature_columns]
    for col in final_df.columns:
        if col.endswith("_encoded"):
            final_df[col] = final_df[col].astype("int64")
        else:
            final_df[col] = final_df[col].astype("bool")

    return final_df


# --- 전처리 함수 정의 끝 ---

# --- 페이지 설정 ---
st.set_page_config(
    page_title="사용자 정보 입력 설문",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- 모델 로드 ---
try:
    model = joblib.load("random_forest_model_sungwoo.pkl")
except FileNotFoundError:
    st.error(
        "Error: Model file 'random_forest_model_sungwoo.pkl' not found. Please ensure the model file is in the correct directory."
    )
    st.stop()
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

selectable_q353_reasons = [
    "방송 못 본 것을 보기 위해서",
    "방송 본 것을 다시 보고 싶어서",
    "이동 시 시간을 활용하기 위해서",
    "광고를 보기 싫어서",
    "원하는 시간에 보기 위해서",
    "TV 수상기에 없는 장소에서 보기 위해서",
    "기타",
]
default_q353_value = "무응답"

# --- 입력 폼 생성 ---
with st.form("survey_form"):
    st.header("👤 기본 인적 사항")
    col1, col2, col3 = st.columns(3)
    with col1:
        dm1_input = st.radio(
            "1. 성별 (dm1)", gender_options, key="dm1", horizontal=True
        )
        # 직업: 랜덤 기본값 설정
        dm4_input = st.selectbox(
            "4. 직업 (DM4)",
            job_options,
            key="dm4",
            index=random.randint(0, len(job_options) - 1),
        )
        # 가구 형태: 랜덤 기본값 설정
        dm8_input = st.selectbox(
            "7. 가구 형태 (dm8)",
            housing_options,
            key="dm8",
            index=random.randint(0, len(housing_options) - 1),
        )
    with col2:
        # 연령: 랜덤 기본값 설정
        DM2_input = st.selectbox(
            "2. 연령 (DM2)",
            age_options,
            key="DM2",
            index=random.randint(0, len(age_options) - 1),
        )
        # 소득: 랜덤 기본값 설정
        DM5_input = st.selectbox(
            "5. 가구 소득 (월 평균) (DM5)",
            income_options,
            key="DM5",
            index=random.randint(0, len(income_options) - 1),
        )
        # 지역 유형: 랜덤 기본값 설정
        DM11_input = st.selectbox(
            "8. 거주 지역 유형 (DM11)",
            city_type_options,
            key="DM11",
            index=random.randint(0, len(city_type_options) - 1),
        )
    with col3:
        # 지역: 랜덤 기본값 설정
        DM3_input = st.selectbox(
            "3. 거주 지역 (DM3)",
            region_options,
            key="DM3",
            index=random.randint(0, len(region_options) - 1),
        )
        # 학력: 랜덤 기본값 설정
        dm6_input = st.selectbox(
            "6. 학력 (dm6)",
            edu_options,
            key="dm6",
            index=random.randint(0, len(edu_options) - 1),
        )

    st.markdown("---")
    st.header("📺 미디어 이용 행태")

    st.markdown(
        "##### 9. 미디어/OTT 서비스 이용 중단 또는 변경 이유 (해당하는 것을 모두 선택)"
    )
    q353_checkbox_states = {}
    reason_cols = st.columns(3)
    col_idx = 0
    for reason in selectable_q353_reasons:
        with reason_cols[col_idx % 3]:
            # 체크박스는 랜덤 기본값 설정이 의미 없음 (True/False)
            q353_checkbox_states[reason] = st.checkbox(reason, key=f"q353_cb_{reason}")
        col_idx += 1

    st.markdown("##### 10. 광고 기반 무료 서비스 이용 의향 (Q531A9)")
    st.caption("예: 광고를 보는 대신 무료로 콘텐츠를 이용하시겠습니까?")
    # 라디오 버튼도 첫 번째 옵션이 기본값으로 선택되므로 랜덤 설정 불필요
    Q531A9_input = st.radio(
        "", q531a9_options, key="Q531A9", horizontal=True, label_visibility="collapsed"
    )

    st.markdown("---")
    submitted = st.form_submit_button("📝 설문 제출 및 예측")  # 버튼 텍스트 변경


# --- 제출 후 처리 ---
if submitted:

    st.markdown("---")

    st.markdown("## 📊 광고 선호도 파생 변수 생성")

    st.markdown(
        """
    ### 📝 파생 변수 설명  
    - **광고 비선호 (`광고 비선호`)**  
    - '광고를 보기 싫어서'라고 응답한 경우 **'비선호'**, 그 외 **'보통'**  
    - **광고 선호 (`광고 선호`)**  
    - '광고를 선호하냐'는 질문(Q531A9)에 **'예'** → **'선호'**  
    - '아니오' → **'보통'**  
    - **상충되는 의견 제거**  
    - '광고 선호'가 '선호'인데 '광고 비선호'가 '비선호'인 경우 제거  
    - **광고 비선호 반영**  
    - '광고 비선호'가 '비선호'이고 '광고 선호'가 '보통'이면 '광고 선호'를 '비선호'로 변경  
    """
    )

    st.markdown("### 🔍 Python 코드")
    st.markdown(
        """
    ```python
    df_melted['광고 비선호'] = df_melted['AD'].apply(lambda x: '비선호' if x == '광고를 보기 싫어서' else '보통')
    df_melted['광고 선호'] = df_melted['Q531A9'].apply(lambda x: '보통' if x == '아니오' else '선호')

    # 광고에 대한 상충되는 의견 제거
    df_melted = df_melted[~((df_melted['광고 선호'] == '선호') & (df_melted['광고 비선호'] == '비선호'))]

    # '광고 선호' 컬럼의 카테고리 목록에 '비선호' 추가
    df_melted['광고 선호'] = df_melted['광고 선호'].cat.add_categories('비선호')

    # 조건에 맞는 행의 '광고 선호' 값을 '비선호'로 변경
    df_melted.loc[(df_melted['광고 비선호'] == '비선호') & (df_melted['광고 선호'] == '보통'), '광고 선호'] = '비선호'
    ```

    """
    )

    st.markdown("---")

    # 설명 추가
    st.markdown(
        """
    ### 🔍 그래프 설명  
    이 그래프는 랜덤 포레스트(Random Forest) 모델의 특성 중요도(Feature Importance)를 나타냅니다.  
    - **X축**: 특성(feature) 중요도 값 (값이 클수록 중요)  
    - **Y축**: 각 특성의 이름  
    - 특성 중요도 값은 랜덤 포레스트가 **결정 트리들을 앙상블하여 학습한 결과**를 바탕으로 산출됨  
    - 높은 중요도를 가진 특성이 모델의 예측에 더 큰 영향을 미침  

    이 정보를 활용하여 **모델 성능 개선** 또는 **불필요한 특성 제거(feature selection)** 등의 작업을 수행할 수 있습니다.  
    특성 중요도를 분석하여 의미 있는 변수만 선택해 모델 최적화를 했습니다다!
    """
    )

    st.image(
        "project_2_image.png",
        caption="랜덤 포레스트 모델의 특성 중요도",
    )
    # 1. 입력값 딕셔너리 생성
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
    selected_reasons = [
        reason for reason, checked in q353_checkbox_states.items() if checked
    ]
    q_cols_outputs = {}
    num_selected = len(selected_reasons)
    for i in range(1, 7):
        col_name = f"Q353M_{i}"
        q_cols_outputs[col_name] = (
            selected_reasons[i - 1] if i <= num_selected else default_q353_value
        )
    survey_data.update(q_cols_outputs)

    # 2. DataFrame 생성 및 표시
    input_df = pd.DataFrame([survey_data])
    st.subheader("✍️ 제출된 설문 내용")
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

    st.success("설문 제출 완료! 데이터 전처리 및 예측을 시작합니다...")
    st.markdown("---")

    # 3. 데이터 전처리
    st.subheader("⚙️ 데이터 전처리 수행")
    try:
        processed_df = load_and_preprocess_team2_data(
            input_df.copy()
        )  # 원본 보존 위해 copy() 사용

        if processed_df is not None:
            st.write("✅ 전처리 완료!")
            # st.dataframe(processed_df) # 전처리 결과는 예측에만 사용하고 숨길 수 있음

            # 4. 모델 예측 수행
            st.markdown("---")
            st.subheader("🔮 모델 예측 결과")
            try:
                # *** 중요: processed_df 컬럼과 모델 학습 시 컬럼 일치 확인 필요 ***
                prediction = model.predict(processed_df)
                prediction_proba = model.predict_proba(processed_df)

                prediction_map = {
                    0: "비선호",
                    1: "보통",
                    2: "선호",
                }  # 0, 1, 2 순서 가정
                predicted_label = prediction_map.get(prediction[0], "알 수 없음")

                # 클래스 순서가 [0, 1, 2] 즉 ['비선호', '보통', '선호'] 라고 가정
                prob_dislike = prediction_proba[0][0] * 100
                prob_neutral = prediction_proba[0][1] * 100
                prob_like = prediction_proba[0][2] * 100

                st.metric(label="예측된 광고 선호도", value=predicted_label)
                st.write("##### 예측 확률:")
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                with prob_col1:
                    st.metric(label="비선호 확률", value=f"{prob_dislike:.1f}%")
                with prob_col2:
                    st.metric(label="보통 확률", value=f"{prob_neutral:.1f}%")
                with prob_col3:
                    st.metric(label="선호 확률", value=f"{prob_like:.1f}%")

                if predicted_label == "선호":
                    st.success("광고 기반 서비스에 긍정적 반응 가능성 높음")
                elif predicted_label == "보통":
                    st.info("광고 기반 서비스에 중립적 반응 가능성")
                else:
                    st.warning("광고 기반 서비스에 부정적 반응 가능성 높음")

            except ValueError as e:
                st.error(
                    f"모델 입력 오류: 전처리 데이터와 모델 학습 데이터의 컬럼(Features)이 일치하지 않습니다. ({e})"
                )
                st.write("모델 입력 데이터 샘플:")
                st.dataframe(processed_df.head(1))  # 실제 입력 데이터 확인
                if hasattr(model, "n_features_in_"):
                    st.write(f"모델이 학습된 Feature 개수: {model.n_features_in_}")
                if hasattr(model, "feature_names_in_"):
                    st.write(
                        f"모델이 학습된 Feature 이름 (일부): {list(model.feature_names_in_[:10])}..."
                    )  # 학습된 컬럼명 확인 (있는 경우)
            except Exception as e:
                st.error(f"예측 중 오류 발생: {e}")
        else:
            st.error("데이터 전처리 중 오류가 발생하여 예측을 수행할 수 없습니다.")
    except Exception as e:
        st.error(f"전처리 함수 실행 중 오류 발생: {e}")
