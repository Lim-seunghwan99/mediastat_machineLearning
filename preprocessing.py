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
