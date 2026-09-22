import streamlit as st
import pandas as pd

# 브라우저 탭 제목과 아이콘 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# 화면 맨 위 제목
st.title("🧠 뇌졸중 예측 실습실")

# ---------------------------
# 데이터 불러오기
# ---------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

st.header("📌 데이터 소개")
st.write(
    "이 데이터는 환자들의 여러 건강 정보를 바탕으로 "
    "뇌졸중(stroke) 발생 여부를 담고 있는 자료입니다."
)

# ---------------------------
# 큰 숫자 카드 네 개
# ---------------------------
total_people = len(df)
total_columns = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = stroke_count / total_people * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 사람 수", f"{total_people:,} 명")
col2.metric("열 개수", f"{total_columns} 개")
col3.metric("뇌졸중(stroke=1) 인원", f"{stroke_count:,} 명")
col4.metric("뇌졸중 비율", f"{stroke_ratio:.2f} %")

st.divider()

# ---------------------------
# 열 이름, 우리말 뜻, 값의 종류, 빈 값 개수 표
# ---------------------------
st.header("📋 열(컬럼) 설명표")
st.caption("※ '우리말 뜻' 칸은 교재를 참고하여 직접 채워 넣어 보세요.")

column_info = []
for col in df.columns:
    dtype = df[col].dtype
    null_count = df[col].isnull().sum()

    # 값의 종류 요약
    if dtype == "object" or df[col].nunique() <= 10:
        unique_vals = df[col].dropna().unique()
        value_type = ", ".join(map(str, unique_vals[:10]))
        if len(unique_vals) > 10:
            value_type += " ..."
    else:
        value_type = f"{df[col].min()} ~ {df[col].max()} (숫자)"

    column_info.append({
        "열 이름": col,
        "우리말 뜻": "",   # 학생이 직접 채우는 칸
        "값의 종류": value_type,
        "빈 값 개수": null_count
    })

info_df = pd.DataFrame(column_info)

# 편집 가능한 표로 제공 (우리말 뜻 칸을 직접 입력 가능)
edited_df = st.data_editor(
    info_df,
    use_container_width=True,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"]  # 우리말 뜻만 수정 가능
)

st.divider()

# ---------------------------
# 데이터 처음 다섯 줄
# ---------------------------
st.header("🔎 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(), use_container_width=True)

st.divider()

# ---------------------------
# 데이터 출처
# ---------------------------
st.header("📚 데이터 출처")
source_text = st.text_area(
    "교재에 적힌 데이터 출처를 이곳에 입력하세요.",
    placeholder="예: 출처 - OOO (20xx), ...",
    height=100
)

if source_text:
    st.success("출처가 입력되었습니다 ✅")
    st.write(source_text)
