import streamlit as st
import pandas as pd
import plotly.express as px

# 브라우저 탭 제목과 아이콘 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 탐색",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 데이터 탐색")

# ---------------------------
# 데이터 불러오기 (첫 화면과 동일)
# ---------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ---------------------------
# 1. 나이와 평균 혈당 분포 (히스토그램 두 개 나란히)
# ---------------------------
st.header("1️⃣ 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df, x="age", nbins=30,
        title="나이(age) 분포",
        labels={"age": "나이"}
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df, x="avg_glucose_level", nbins=30,
        title="평균 혈당(avg_glucose_level) 분포",
        labels={"avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose, use_container_width=True)

st.divider()

# ---------------------------
# 2. 뇌졸중 여부에 따른 나이/평균 혈당 상자그림 + 평균값 표
# ---------------------------
st.header("2️⃣ 뇌졸중 여부에 따른 나이·평균 혈당 비교")

# stroke 값을 문자열로 바꿔서 그래프에 표시 (0 -> 없음, 1 -> 있음)
df_box = df.copy()
df_box["stroke_label"] = df_box["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

col3, col4 = st.columns(2)

with col3:
    fig_box_age = px.box(
        df_box, x="stroke_label", y="age",
        title="뇌졸중 여부에 따른 나이 비교",
        labels={"stroke_label": "뇌졸중 여부", "age": "나이"}
    )
    st.plotly_chart(fig_box_age, use_container_width=True)

with col4:
    fig_box_glucose = px.box(
        df_box, x="stroke_label", y="avg_glucose_level",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_box_glucose, use_container_width=True)

# 평균값 표
st.subheader("📊 두 그룹의 평균값")
mean_table = df_box.groupby("stroke_label")[["age", "avg_glucose_level"]].mean().round(2)
mean_table.columns = ["나이 평균", "평균 혈당 평균"]
st.dataframe(mean_table, use_container_width=True)

st.divider()

# ---------------------------
# 3. 고혈압/심장병 유무에 따른 뇌졸중 비율 막대그래프
# ---------------------------
st.header("3️⃣ 고혈압·심장병 유무에 따른 뇌졸중 비율")

col5, col6 = st.columns(2)

with col5:
    hyper_ratio = df.groupby("hypertension")["stroke"].mean().reset_index()
    hyper_ratio["hypertension_label"] = hyper_ratio["hypertension"].map({0: "고혈압 없음", 1: "고혈압 있음"})
    hyper_ratio["stroke_percent"] = hyper_ratio["stroke"] * 100

    fig_hyper = px.bar(
        hyper_ratio, x="hypertension_label", y="stroke_percent",
        title="고혈압 유무에 따른 뇌졸중 비율",
        labels={"hypertension_label": "고혈압 여부", "stroke_percent": "뇌졸중 비율(%)"},
        text=hyper_ratio["stroke_percent"].round(2)
    )
    st.plotly_chart(fig_hyper, use_container_width=True)

with col6:
    heart_ratio = df.groupby("heart_disease")["stroke"].mean().reset_index()
    heart_ratio["heart_disease_label"] = heart_ratio["heart_disease"].map({0: "심장병 없음", 1: "심장병 있음"})
    heart_ratio["stroke_percent"] = heart_ratio["stroke"] * 100

    fig_heart = px.bar(
        heart_ratio, x="heart_disease_label", y="stroke_percent",
        title="심장병 유무에 따른 뇌졸중 비율",
        labels={"heart_disease_label": "심장병 여부", "stroke_percent": "뇌졸중 비율(%)"},
        text=heart_ratio["stroke_percent"].round(2)
    )
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# ---------------------------
# 4. bmi 결측치 그룹과 전체의 뇌졸중 비율 비교
# ---------------------------
st.header("4️⃣ 체질량지수(bmi) 결측치와 뇌졸중 비율")

bmi_missing_df = df[df["bmi"].isnull()]
bmi_missing_count = len(bmi_missing_df)
bmi_missing_stroke_ratio = bmi_missing_df["stroke"].mean() * 100 if bmi_missing_count > 0 else 0
overall_stroke_ratio = df["stroke"].mean() * 100

compare_table = pd.DataFrame({
    "구분": ["bmi 결측치 그룹", "전체 데이터"],
    "사람 수": [bmi_missing_count, len(df)],
    "뇌졸중 비율(%)": [round(bmi_missing_stroke_ratio, 2), round(overall_stroke_ratio, 2)]
})

st.dataframe(compare_table, use_container_width=True)

st.divider()

# ---------------------------
# 5. 흡연 상태별 사람 수 표
# ---------------------------
st.header("5️⃣ 흡연 상태(smoking_status)별 사람 수")

smoking_count = df["smoking_status"].value_counts().reset_index()
smoking_count.columns = ["흡연 상태", "사람 수"]

st.dataframe(smoking_count, use_container_width=True)
