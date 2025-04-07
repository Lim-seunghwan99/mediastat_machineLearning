import streamlit as st
import pandas as pd

# --- 페이지 설정 ---
st.set_page_config(
    page_title="동영상 카테고리 추천 AI - 소개",  # 페이지 제목 명확화
    page_icon="🎬",  # 페이지 아이콘 변경
    layout="wide",
    initial_sidebar_state="auto",
)

# --- 헤더 섹션 ---
st.title("🎬 맞춤 동영상 카테고리 추천 AI")
st.markdown(
    "#### **설문 데이터**를 분석하여 당신에게 **최적화된 동영상 카테고리**를 찾아드립니다."
)
# 부가 정보는 캡션이나 작은 글씨로
st.caption("메타버스 아카데미 | P-TYPE 팀 | 머신러닝 프로젝트")
st.markdown("---")  # 구분선

# --- 프로젝트 소개 섹션 ---
st.header("🚀 프로젝트 개요")
st.write("")  # 약간의 공백 추가

col1, col2 = st.columns(2)  # 2개의 컬럼 생성

with col1:
    st.subheader("🎯 프로젝트 주제")
    st.markdown(
        """
        본 프로젝트는 사용자의 **인구통계학적 정보, 스마트폰 이용 행태, 미디어 소비 습관** 등
        다양한 설문 응답 데이터를 머신러닝 모델로 분석합니다.

        이를 통해 개개인의 **숨겨진 선호도를 파악**하고, 가장 흥미를 느낄 만한
        **동영상 콘텐츠 카테고리** (예: 스포츠, 웹예능, 푸드, 뷰티 등)를
        **맞춤형으로 추천**하는 것을 목표로 합니다.
        """
    )

with col2:
    st.subheader("💡 프로젝트 동기")
    st.markdown(
        """
        **방송통신위원회**에서 제공하는 **'방송매체 이용행태 조사' 공공데이터**의
        활용 가능성에 주목했습니다.

        이 데이터를 기반으로 사용자들이 방대한 영상 콘텐츠의 홍수 속에서
        **자신의 취향에 맞는 영상을 더 쉽고 빠르게 발견**하도록 돕는,
        **실질적이고 유용한 서비스**를 만들고자 본 프로젝트를 기획하게 되었습니다.

        *데이터 출처: [방송통신위원회](https://www.kcc.go.kr) 방송매체 이용행태 조사*
        """
    )

st.markdown("---")  # 구분선

# --- 프로젝트 정보 섹션 ---
st.header("ℹ️ 프로젝트 정보")

# 세부 정보는 Expander 안에 넣어서 깔끔하게 관리

st.markdown(
    f"""
        *   **수행 기관:** 메타버스 아카데미
        *   **팀 명:** P-TYPE
        *   **🗓️ 프로젝트 기간:** 2025년 03월 31일 ~ 2025년 04월 04일
        *   **👥 팀원:** 박정훈, 배성우, 임승환
        """
)


st.markdown("---")  # 구분선

st.write("")  # Expander와의 간격 조절
st.subheader("📊 사용 데이터")
st.markdown(
    """
    이 프로젝트는 **방송통신위원회 방송통계포털**에서 제공하는 데이터를 기반으로 합니다.

    *   **데이터 명:** 2022년 방송매체 이용행태 조사
    *   **데이터 출처:** [방송통계포털 (mediastat.or.kr)](https://www.mediastat.or.kr/kor/contents/ContentsList.html)
    """
)


# (기존 코드 마지막 부분)
st.markdown(
    """
    이 프로젝트는 **방송통신위원회 방송통계포털**에서 제공하는 데이터를 기반으로 합니다.

    *   **데이터 명:** 2022년 방송매체 이용행태 조사
    *   **데이터 출처:** [방송통계포털 (mediastat.or.kr)](https://www.mediastat.or.kr/kor/contents/ContentsList.html)
    """
)

# --- 모델 학습 알고리즘 섹션 추가 ---
st.markdown("---")  # 구분선 추가
st.subheader("⚙️ 모델 학습 알고리즘")
st.markdown(
    """
    사용자에게 **하나 이상의 관련 동영상 카테고리를 추천**하는 **다중 레이블 분류(Multi-label Classification)** 문제를 해결하기 위해 다음과 같은 알고리즘을 조합하여 사용했습니다.

    *   **기본 분류기: `XGBClassifier` (XGBoost)**
        *   XGBoost 자체는 기본적으로 다중 클래스 분류를 지원하지만, 다중 레이블 분류는 직접 지원하지 않습니다.
        *   따라서 다중 레이블 분류를 위해 `OneVsRestClassifier`와 함께 사용했습니다. 

    *   `OneVsRestClassifier`란?
        *   하나의 분류기 모델(여기서는 XGBoost)을 각 카테고리(레이블)마다 개별적으로 학습시키는 전략입니다.
        *   예를 들어 '스포츠'를 좋아하는지 아닌지 예측하는 모델, '웹예능'을 좋아하는지 아닌지 예측하는 모델 등을 각각 독립적으로 만듭니다.
        *   이를 통해 각 사용자가 **여러 개의 카테고리에 동시에 해당될 수 있는** 다중 레이블 예측이 가능해집니다.
    
    * 둘을 조합했을때 장점
        *   각 레이블에 대해 개별 이진 분류기를 학습하여 정확한 예측 가능
        *   클래스 불균형이 심할 때 유리

    이 조합을 통해 사용자의 복합적인 선호도를 반영하여 여러 개의 동영상 카테고리를 효과적으로 추천했습니다.
    """
)


# --- 모델 최적화 섹션 추가 ---
st.markdown("---")  # 구분선 추가
st.subheader("🛠️ 모델 최적화")
st.markdown(
    """
    모델의 성능을 향상시키기 위해 다음과 같은 최적화 과정을 수행했습니다.

    **1. 클래스 불균형 해소:**
    *   다중 레이블 데이터셋에서는 특정 카테고리(레이블)를 선호하는 사용자가 다른 카테고리에 비해 매우 적거나 많을 수 있습니다 (클래스 불균형).
    *   이러한 불균형은 모델이 소수 카테고리를 잘 예측하지 못하게 만들 수 있습니다.
    *   `XGBClassifier`의 `sample_weight="balanced"` 옵션을 사용하여 학습 시 각 클래스(카테고리)의 중요도를 데이터 양에 반비례하도록 조정하여 불균형 문제를 완화하고자 했습니다.

    ```python
    from sklearn.multiclass import OneVsRestClassifier
    from xgboost import XGBClassifier

    XGBmodel = OneVsRestClassifier(XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42,
                                    scale_pos_weight=... # 또는 scale_pos_weight 파라미터 활용
    ))
    ```


    **2. 하이퍼파라미터 튜닝 (Grid Search):**
    *   모델의 성능은 `n_estimators`, `learning_rate`, `max_depth` 등 다양한 하이퍼파라미터 설정에 크게 영향을 받습니다.
    *   `GridSearchCV`를 사용하여 미리 정의된 하이퍼파라미터 조합들을 교차 검증(Cross-Validation) 방식으로 평가하고, 가장 좋은 성능(여기서는 F1-Micro 점수 기준)을 내는 최적의 조합을 탐색했습니다.
    *   `OneVsRestClassifier`와 함께 사용하기 위해 파라미터 이름 앞에 `estimator__`를 붙여 XGBoost 내부 파라미터를 지정했습니다.

    ```python
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from sklearn.multiclass import OneVsRestClassifier

    # 탐색할 하이퍼파라미터 그리드 설정
    param_grid = {
        'estimator__n_estimators': [300, 400, 500],
        'estimator__learning_rate': [0.05, 0.1, 0.2],
        'estimator__max_depth': [4, 5, 6],
        'estimator__min_child_weight': [1, 2, 3],
        'estimator__subsample': [0.75, 0.85, 0.95],
        'estimator__colsample_bytree': [0.7, 0.8, 0.9]
    }

    # GridSearchCV 설정 (OneVsRestClassifier 래핑)
    base_model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss', random_state=42))

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='f1_micro', # 다중 레이블 평가지표
        cv=3, # 3-Fold 교차 검증
        verbose=2,
        n_jobs=-1 # 사용 가능한 모든 CPU 코어 사용
    )

    ```
    이러한 최적화 과정을 통해 모델의 일반화 성능과 예측 정확도를 높이고자 노력했습니다.
    """
)


st.markdown("---")  # 구분선 추가
st.subheader("🔄 데이터 전처리")
st.markdown(
    """
        머신러닝 모델은 숫자 형태의 데이터를 입력으로 받기 때문에, 설문 조사의 문자열 응답들을 적절한 숫자 형태로 변환하는 전처리 과정이 필수적입니다. 본 프로젝트에서는 다음과 같은 인코딩 방법을 주로 사용했습니다.
        
        **1. 레이블 인코딩 (Label Encoding):**
        *   카테고리 값에 순서가 중요하지 않은 명목형 변수(Nominal Variable)에 주로 적용했습니다
        
        **2. 순서형 인코딩 (Ordinal Encoding):**
        *   카테고리 값 간에 명확한 순서가 있는 서열 변수(Ordinal Variable)에 적용했습니다.
    """
)
with st.expander("클릭하면 전처리 함수를 확인할 수 있습니다.", expanded=False):
    st.markdown(
        """

            ```python
            # preprocessing.py

            from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


            def preprocess_data(df):
                print(df.columns)  # 현재 컬럼 확인

                # ✅ 라벨 인코딩할 컬럼
                label_encoding_columns = ["dm1", "DM2", "DM3", "DM4", "dm7"]

                # ✅ 순서형 인코딩할 컬럼과 카테고리 정의
                ordinal_categories = {
                    "dm6": ["중/고등학생", "고졸이하", "대학생/대학원생", "대졸이상"],
                    "dm8": ["독신가구", "1세대가구", "2세대가구", "3세대가구", "기타"],
                    "Q417": [
                        "3만원 미만",
                        "3만원 이상-4만원 미만",
                        "4만원 이상-5만원 미만",
                        "5만원 이상-6만원 미만",
                        "6만원 이상-7만원 미만",
                        "7만원 이상-10만원 미만",
                        "10만원 이상",
                    ],
                }

                ordinal_group_categories = {
                    "Q424A": [
                        "전혀 중요하지 않다",
                        "중요하지 않다",
                        "보통이다",
                        "중요하다",
                        "매우 중요하다",
                    ],
                    "Q419A": [
                        "전혀안봄/이용안함",
                        "2-3달에 1-2일 이하",
                        "한달에 1-3일",
                        "일주일에 1-2일",
                        "일주일에 3-4일",
                        "일주일에 5-6일",
                        "매일",
                    ],
                    "Q263A": [
                        "전혀 그렇지 않다",
                        "그렇지 않은 편이다",
                        "보통이다",
                        "그런 편이다",
                        "매우 그렇다",
                    ],
                }

                # ✅ 모든 컬럼을 자동으로 확인하여 처리
                for col in df.columns:
                    # 🎯 라벨 인코딩
                    if col in label_encoding_columns:
                        print(f"라벨인코딩 {col} 전처리 시작")
                        encoder = LabelEncoder()
                        df[f"{col}_encoded"] = encoder.fit_transform(df[col])
                        df.drop(columns=[col], inplace=True)  # 원본 컬럼 삭제

                    # 🎯 순서형 인코딩 (단일 컬럼)
                    elif col in ordinal_categories:
                        print(f"순서형 인코딩 {col} 전처리 시작")
                        encoder = OrdinalEncoder(
                            categories=[ordinal_categories[col]]
                        )  # 차원 맞추기
                        df[f"{col}_encoded"] = encoder.fit_transform(df[[col]])
                        df.drop(columns=[col], inplace=True)  # 원본 컬럼 삭제

                # ✅ Q424A1 ~ Q424A4, Q419A1 ~ Q419A8, Q263A1 ~ Q263A5 개별 처리
                for prefix, categories in ordinal_group_categories.items():
                    print(f"{prefix}, {categories} 전처리 시작")
                    target_cols = [
                        col for col in df.columns if col.startswith(prefix)
                    ]  # 관련 컬럼 찾기
                    for col in target_cols:  # 🎯 각 컬럼별 개별 인코딩 적용
                        encoder = OrdinalEncoder(categories=[categories])  # 차원 맞추기
                        df[f"{col}_encoded"] = encoder.fit_transform(df[[col]])
                        df.drop(columns=[col], inplace=True)  # 원본 컬럼 삭제

                return df
            ```
        """
    )


st.markdown("---")  # 구분선 추가
st.subheader("📊 모델 성능 평가 (F1 Score)")
st.markdown(
    """
    모델이 각 동영상 카테고리를 얼마나 잘 예측하는지 평가하기 위해 **F1 Score**를 주요 지표로 활용했습니다.

    **🤔 F1 Score란?**
    *   모델의 정밀도(Precision)와 재현율(Recall)의 조화 평균(Harmonic Mean)입니다.
    *   **정밀도(Precision):** 모델이 특정 카테고리라고 예측한 것 중, 실제로 해당 카테고리를 선호하는 사용자의 비율입니다. (예: 모델이 '스포츠'라고 추천했을 때, 실제로 사용자가 '스포츠'를 좋아할 확률) - *얼마나 정확하게 예측했는가?*
    *   **재현율(Recall):** 실제로 특정 카테고리를 선호하는 사용자 중, 모델이 해당 카테고리라고 예측한 사용자의 비율입니다. (예: 실제로 '스포츠'를 좋아하는 사용자 중, 모델이 '스포츠'라고 맞춘 확률) - *얼마나 빠짐없이 찾아냈는가?*
    *   F1 Score는 정밀도와 재현율이 모두 중요할 때 사용되며, 특히 데이터 불균형(어떤 카테고리는 인기가 많고 어떤 카테고리는 적은 경우)이 있을 때 유용합니다. 값이 1에 가까울수록 모델 성능이 좋음을 의미합니다.

    아래 그래프는 학습된 모델이 **각 동영상 카테고리별로** 어느 정도의 F1 Score를 보이는지 시각화한 결과입니다. 이를 통해 어떤 카테고리를 상대적으로 잘 예측하고, 어떤 카테고리 예측에 어려움이 있는지 파악할 수 있습니다.
    """
)

# 이미지 파일 로드 및 표시 (이미지 파일이 Streamlit 앱 파일과 같은 디렉토리에 있다고 가정)
try:
    st.image(
        "f1_score_plot.png",
        caption="각 동영상 카테고리별 F1 Score",
    )
    st.markdown(
        """
        *그래프의 각 막대는 해당 카테고리에 대한 모델의 예측 성능(F1 Score)을 나타냅니다. 막대가 높을수록 해당 카테고리에 대한 예측이 더 정확하고 빠짐없음을 의미합니다.*
        """
    )
except FileNotFoundError:
    st.error(
        "⚠️ 'f1_score_plot.png' 파일을 찾을 수 없습니다. Streamlit 앱과 같은 디렉토리에 있는지 확인해주세요."
    )
except Exception as e:
    st.error(f"이미지를 로드하는 중 오류가 발생했습니다: {e}")
