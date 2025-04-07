# preprocessing_team2.py

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def load_and_preprocess_team2_data(df):
    # 1. 데이터 불러오기

    # 2. 주요 컬럼 선택
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
    df = df[save_cols].dropna(subset=["Q531A9"]).reset_index(drop=True)

    # 3. 컬럼명 변경
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
    df.rename(columns=rename_map, inplace=True)

    # 4. Q353M_1~6 melt
    q353_cols = [f"Q353M_{i}" for i in range(1, 7)]
    df_melted = df.melt(
        id_vars=list(rename_map.values()) + ["Q531A9"],
        value_vars=q353_cols,
        var_name="Q353",
        value_name="AD",
    )

    # 5. 광고 비선호/선호 파생
    df_melted["광고비선호"] = df_melted["AD"].apply(
        lambda x: "비선호" if x == "광고를 보기 싫어서" else "보통"
    )
    df_melted["광고선호"] = df_melted["Q531A9"].apply(
        lambda x: "보통" if x == "아니오" else "선호"
    )

    # 6. 충돌 제거
    df_melted = df_melted[
        ~((df_melted["광고선호"] == "선호") & (df_melted["광고비선호"] == "비선호"))
    ]

    # 7. 광고 선호 정제
    df_melted["광고선호"] = df_melted["광고선호"].astype("category")
    if "비선호" not in df_melted["광고선호"].cat.categories:
        df_melted["광고선호"] = df_melted["광고선호"].cat.add_categories(["비선호"])
    df_melted.loc[
        (df_melted["광고비선호"] == "비선호") & (df_melted["광고선호"] == "보통"),
        "광고선호",
    ] = "비선호"

    # 8. 불필요 컬럼 제거
    df_cleaned = df_melted.drop(
        columns=["Q531A9", "Q353", "AD", "광고비선호"]
    ).reset_index(drop=True)

    # 9. 소득 무응답 제거
    df_cleaned = df_cleaned[df_cleaned["소득"] != "무응답"]

    # 10. 순서형 인코딩
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

    for col, categories in ordinal_map.items():
        encoder = OrdinalEncoder(categories=[categories])
        df_cleaned[f"{col}_encoded"] = encoder.fit_transform(df_cleaned[[col]])

    # 11. 원핫 인코딩
    onehot_cols = ["성별", "지역", "직업", "가구형태", "도시유형"]
    df_encoded = pd.get_dummies(df_cleaned, columns=onehot_cols)

    return df_encoded
