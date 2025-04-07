import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import streamlit as st  # Streamlit 에러 메시지 표시용 (선택 사항)


def load_and_preprocess_team2_data(input_df):
    """
    Streamlit에서 입력받은 단일 행 DataFrame을 전처리하여
    지정된 27개 feature 컬럼 구조로 반환합니다.

    Args:
        input_df (pd.DataFrame): Streamlit 입력으로 생성된 DataFrame.
                                 컬럼명은 원본 SPSS 컬럼명 (dm1, DM2 등)이어야 함.

    Returns:
        pd.DataFrame: 전처리 완료된 DataFrame (27개 feature 컬럼) 또는 오류 시 None.
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
    # 입력 df에 필요한 컬럼이 모두 있는지 확인 (선택적이지만 권장)
    missing_cols = [col for col in save_cols if col not in input_df.columns]
    if missing_cols:
        st.error(
            f"입력 데이터에 필요한 컬럼이 누락되었습니다: {', '.join(missing_cols)}"
        )
        return None

    # 작업할 복사본 생성
    df_processed = input_df[save_cols].copy()

    # 1. 컬럼명 변경 (원본 함수의 Step 3)
    rename_map = {
        "dm1": "성별",
        "DM2": "나이대",
        "DM3": "지역",
        "DM4": "직업",
        "DM5": "소득",
        "dm6": "학력",
        "dm8": "가구형태",
        "DM11": "도시유형",  # 오타 수정: DM11
    }
    df_processed.rename(columns=rename_map, inplace=True)

    # 2. '광고비선호' 상태 확인 (원본 함수의 Step 4, 5 Melt 로직 대체)
    is_ad_averse = False
    q353_cols_original = [f"Q353M_{i}" for i in range(1, 7)]
    for col in q353_cols_original:
        # .iloc[0] 사용하여 첫 번째 (유일한) 행의 값 확인
        if df_processed.loc[0, col] == "광고를 보기 싫어서":
            is_ad_averse = True
            break

    # 3. '광고선호' 상태 결정 (원본 함수의 Step 5)
    q531a9_val = df_processed.loc[0, "Q531A9"]
    # '모름/무응답'도 '보통'으로 처리 (원본 로직 유추 또는 정책 결정 필요)
    if q531a9_val in ["아니오", "모름/무응답"]:
        initial_ad_preference = "보통"
    elif q531a9_val == "예":
        initial_ad_preference = "선호"
    else:
        initial_ad_preference = "보통"  # 예상치 못한 값 처리

    # 5. 최종 '광고선호' 결정 (원본 함수의 Step 7)
    if is_ad_averse and initial_ad_preference == "보통":
        final_ad_preference = "비선호"
    else:
        final_ad_preference = initial_ad_preference
    # 이 값을 가진 컬럼을 잠시 추가 (나중에 인코딩 후 제거)
    df_processed["광고선호"] = final_ad_preference

    # 6. 소득 '무응답' 확인 (원본 함수의 Step 9)
    if df_processed.loc[0, "소득"] == "무응답":
        st.error("입력 오류: '소득'은 '무응답'일 수 없습니다.")
        return None

    # 7. 순서형 인코딩 (원본 함수의 Step 10)
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
        "광고선호": ["비선호", "보통", "선호"],  # Target 변수도 일단 인코딩
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
                ).astype(
                    "int64"
                )  # int64로 변환
                ordinal_encoded_cols.append(encoded_col_name)  # 인코딩된 컬럼 이름 저장
            except ValueError as e:
                st.error(
                    f"'{col}' 컬럼 순서형 인코딩 오류: 입력값 '{df_processed.loc[0, col]}' 확인 필요. ({e})"
                )
                return None
        else:
            st.warning(
                f"Warning: 순서형 인코딩 대상 컬럼 '{col}'이(가) 데이터에 없습니다."
            )

    # 8. 원핫 인코딩 (원본 함수의 Step 11)
    onehot_cols = ["성별", "지역", "직업", "가구형태", "도시유형"]
    # bool 타입으로 원핫 인코딩 수행
    df_processed = pd.get_dummies(
        df_processed, columns=onehot_cols, drop_first=False, dtype=bool
    )

    # 9. 최종 컬럼 선택 및 생성/정리
    target_feature_columns = [
        "나이대_encoded",
        "소득_encoded",
        "학력_encoded",  # 순서형 인코딩 결과
        # 원-핫 인코딩 결과 (알파벳/가나다 순 정렬 가정 - pd.get_dummies 기본 동작)
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
        "도시유형_중소도시",  # DM11 -> 도시유형, 원본에 오타 수정 반영
    ]

    # 최종 결과 DataFrame 생성 준비
    final_df = pd.DataFrame()

    # 필요한 컬럼이 df_processed에 있는지 확인하고, 없으면 False(0)으로 채우기
    for col in target_feature_columns:
        if col in df_processed.columns:
            final_df[col] = df_processed[col]
        elif col.endswith("_encoded"):  # 순서형 컬럼이 없으면 에러 가능성 높음
            st.error(f"오류: 필요한 순서형 인코딩 컬럼 '{col}'이 생성되지 않았습니다.")
            return None
        else:  # 원핫 인코딩 컬럼이 없는 경우 (해당 카테고리가 입력되지 않음)
            final_df[col] = False  # bool 타입으로 0 채우기

    # 컬럼 순서를 target_feature_columns 순서대로 맞춤
    final_df = final_df[target_feature_columns]

    # 데이터 타입 최종 확인 및 변환 (필요시)
    for col in final_df.columns:
        if col.endswith("_encoded"):
            final_df[col] = final_df[col].astype("int64")
        else:
            final_df[col] = final_df[col].astype("bool")

    return final_df
