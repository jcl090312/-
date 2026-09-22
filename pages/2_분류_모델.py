import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 분류 모델 만들기")

# ---------------------------
# 데이터 불러오기
# ---------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# 열 이름 <-> 우리말 이름 매핑
col_to_kor = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
kor_to_col = {v: k for k, v in col_to_kor.items()}

feature_options = list(col_to_kor.values())
default_features = ["나이", "평균 혈당", "고혈압", "심장병"]  # bmi 제외

# ---------------------------
# 속성 선택
# ---------------------------
st.header("1️⃣ 입력으로 사용할 속성 고르기")

selected_kor = st.multiselect(
    "모델에 사용할 속성을 고르세요 (처음에는 체질량지수를 뺀 넷이 선택되어 있습니다)",
    options=feature_options,
    default=default_features
)

if len(selected_kor) < 2:
    st.warning("⚠️ 속성을 두 개 이상 골라야 합니다. 속성을 더 선택해 주세요.")
    st.stop()

selected_cols = [kor_to_col[k] for k in selected_kor]
use_bmi = "bmi" in selected_cols

st.divider()

# ---------------------------
# 데이터 준비: 10명씩 묶어 앞 3명을 테스트로 고정
# ---------------------------
st.header("2️⃣ 데이터 나누기")

df_sorted = df.sort_values("id").reset_index(drop=True)
df_sorted["group_index"] = df_sorted.index % 10

test_mask = df_sorted["group_index"] < 3
train_mask = ~test_mask

train_df = df_sorted[train_mask].copy()
test_df = df_sorted[test_mask].copy()

st.write(f"훈련용 사람 수: **{len(train_df):,} 명**, 테스트용 사람 수: **{len(test_df):,} 명**")

# bmi 결측치 처리: 훈련용 중앙값으로 채우기 (bmi를 선택했을 때만)
if use_bmi:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)
    st.write(f"체질량지수(bmi)의 빈 값은 훈련용 중앙값 **{bmi_median:.2f}** 로 채웠습니다.")

X_train = train_df[selected_cols]
y_train = train_df["stroke"]
X_test = test_df[selected_cols]
y_test = test_df["stroke"]

st.divider()

# ---------------------------
# 모델 학습
# ---------------------------
st.header("3️⃣ 모델 학습과 정확도")

# 로지스틱 회귀
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# 의사결정트리 (질문 3번까지, 마지막 마디 5명 미만이면 그만, 난수 고정)
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)
tree_model.fit(X_train, y_train)

# 더미 모델 (많은 쪽으로만 답함)
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)

def get_accuracies(model):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    return train_acc, test_acc

log_train_acc, log_test_acc = get_accuracies(log_model)
tree_train_acc, tree_test_acc = get_accuracies(tree_model)
dummy_train_acc, dummy_test_acc = get_accuracies(dummy_model)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("로지스틱 회귀(확률로 답하는 모델) 테스트 정확도", f"{log_test_acc:.3f}")
    st.caption(f"훈련 정확도: {log_train_acc:.3f} · 테스트 정확도: {log_test_acc:.3f}")

with col2:
    st.metric("의사결정트리(질문으로 답하는 모델) 테스트 정확도", f"{tree_test_acc:.3f}")
    st.caption(f"훈련 정확도: {tree_train_acc:.3f} · 테스트 정확도: {tree_test_acc:.3f}")

with col3:
    st.metric("항상 많은 쪽으로만 답하는 모델 테스트 정확도", f"{dummy_test_acc:.3f}")
    st.caption(f"훈련 정확도: {dummy_train_acc:.3f} · 테스트 정확도: {dummy_test_acc:.3f}")

st.divider()

# ---------------------------
# 산점도: 두 축 고르기
# ---------------------------
st.header("4️⃣ 두 속성으로 그려보는 분류 결과")

if len(selected_kor) < 2:
    st.warning("⚠️ 속성을 두 개 이상 골라야 그림을 그릴 수 있습니다.")
    st.stop()

col_x, col_y = st.columns(2)
with col_x:
    x_kor = st.selectbox("가로축(X축)으로 사용할 속성", selected_kor, index=0)
with col_y:
    remaining = [k for k in selected_kor if k != x_kor]
    y_kor = st.selectbox("세로축(Y축)으로 사용할 속성", remaining, index=0)

x_col = kor_to_col[x_kor]
y_col = kor_to_col[y_kor]

other_cols = [c for c in selected_cols if c not in [x_col, y_col]]

# 두 축이 아닌 속성은 테스트 데이터 중앙값으로 고정
fixed_values = {}
for c in other_cols:
    fixed_values[c] = X_test[c].median()

if fixed_values:
    fixed_text = ", ".join([f"{col_to_kor[c]} = {v:.2f}" for c, v in fixed_values.items()])
    st.write(f"📌 그림에 나타나지 않는 속성은 테스트 데이터의 중앙값으로 고정했습니다: **{fixed_text}**")
else:
    st.write("📌 선택한 속성이 두 개뿐이라 고정할 속성이 없습니다.")

# ---------------------------
# 결정 경계(로지스틱 회귀, 0.5 기준선) 및 결정트리 영역 계산
# ---------------------------
x_min, x_max = X_test[x_col].min(), X_test[x_col].max()
y_min, y_max = X_test[y_col].min(), X_test[y_col].max()

x_range = np.linspace(x_min, x_max, 200)
y_range = np.linspace(y_min, y_max, 200)
xx, yy = np.meshgrid(x_range, y_range)

# 그리드용 데이터프레임 만들기 (선택한 모든 속성 순서 맞추기)
grid_df = pd.DataFrame({x_col: xx.ravel(), y_col: yy.ravel()})
for c in other_cols:
    grid_df[c] = fixed_values[c]
grid_df = grid_df[selected_cols]  # 학습 때와 같은 열 순서로 맞추기

# 결정트리 영역 예측 (배경색용)
tree_grid_pred = tree_model.predict(grid_df).reshape(xx.shape)

# 로지스틱 회귀 확률 예측 (0.5 경계선용)
log_grid_prob = log_model.predict_proba(grid_df)[:, 1].reshape(xx.shape)

fig = go.Figure()

# 결정트리 영역을 옅은 색으로 표시
fig.add_trace(go.Contour(
    x=x_range, y=y_range, z=tree_grid_pred,
    showscale=False,
    colorscale=[[0, "rgba(99,110,250,0.15)"], [1, "rgba(239,85,59,0.15)"]],
    contours=dict(start=0, end=1, size=1),
    name="의사결정트리 영역",
    hoverinfo="skip"
))

# 로지스틱 회귀 0.5 경계선
fig.add_trace(go.Contour(
    x=x_range, y=y_range, z=log_grid_prob,
    showscale=False,
    contours=dict(start=0.5, end=0.5, size=0.1, coloring="lines"),
    line=dict(width=3, color="black"),
    name="로지스틱 회귀 0.5 기준선",
    hoverinfo="skip"
))

# 경계선이 그림 안에 들어오는지 확인
boundary_exists = (log_grid_prob.min() < 0.5) and (log_grid_prob.max() > 0.5)
if not boundary_exists:
    st.info("📌 로지스틱 회귀의 0.5 기준선은 이 그림의 범위 밖에 있어 화면에 나타나지 않습니다.")

# 테스트 데이터 점 찍기 (실제 뇌졸중 여부로 색 구분)
test_plot_df = test_df.copy()
test_plot_df["실제 뇌졸중 여부"] = test_plot_df["stroke"].map({0: "뇌졸중 아님", 1: "뇌졸중"})

for label, color in [("뇌졸중 아님", "blue"), ("뇌졸중", "red")]:
    subset = test_plot_df[test_plot_df["실제 뇌졸중 여부"] == label]
    fig.add_trace(go.Scatter(
        x=subset[x_col], y=subset[y_col],
        mode="markers",
        marker=dict(color=color, size=6, opacity=0.6),
        name=label
    ))

fig.update_layout(
    title=f"{x_kor} vs {y_kor} — 테스트 데이터와 분류 경계",
    xaxis_title=x_kor,
    yaxis_title=y_kor,
    legend_title="실제 뇌졸중 여부"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------
# 의사결정트리 가지 그림 (Graphviz DOT)
# ---------------------------
st.header("5️⃣ 의사결정트리가 던진 질문")

def build_tree_dot(tree, feature_names_kor):
    tree_ = tree.tree_
    dot_lines = ["digraph Tree {", 'node [shape=box, style="filled", fontname="Malgun Gothic"];']

    def recurse(node_id):
        n_samples = tree_.n_node_samples[node_id]
        # value: [클래스0 개수, 클래스1 개수]
        value = tree_.value[node_id][0]
        n_stroke = int(value[1])
        ratio = n_stroke / n_samples if n_samples > 0 else 0

        is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]

        if is_leaf:
            pred_class = int(np.argmax(value))
            pred_label = "뇌졸중" if pred_class == 1 else "아님"
            color = "#f5a3a3" if pred_class == 1 else "#a3c9f5"
            label = f"답: {pred_label}\\n인원 {n_samples}명\\n뇌졸중 {n_stroke}명\\n비율 {ratio:.2f}"
            dot_lines.append(f'{node_id} [label="{label}", fillcolor="{color}"];')
        else:
            feature_idx = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]
            feature_name = feature_names_kor[feature_idx]
            label = (f"{feature_name} <= {threshold:.2f} ?\\n"
                     f"인원 {n_samples}명\\n뇌졸중 {n_stroke}명\\n비율 {ratio:.2f}")
            dot_lines.append(f'{node_id} [label="{label}", fillcolor="#ffffff"];')

            left_id = tree_.children_left[node_id]
            right_id = tree_.children_right[node_id]

            recurse(left_id)
            recurse(right_id)

            dot_lines.append(f'{node_id} -> {left_id} [label="예"];')
            dot_lines.append(f'{node_id} -> {right_id} [label="아니요"];')

    recurse(0)
    dot_lines.append("}")
    return "\n".join(dot_lines)

feature_names_kor_list = [col_to_kor[c] for c in selected_cols]
dot_string = build_tree_dot(tree_model, feature_names_kor_list)

st.graphviz_chart(dot_string)

# ---------------------------
# 답을 내는 마디(리프) 요약
# ---------------------------
tree_ = tree_model.tree_
leaf_count = 0
no_stroke_leaf_count = 0

for node_id in range(tree_.node_count):
    is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]
    if is_leaf:
        leaf_count += 1
        value = tree_.value[node_id][0]
        pred_class = int(np.argmax(value))
        if pred_class == 0:
            no_stroke_leaf_count += 1

# 실제로 트리가 사용한 속성 찾기
used_feature_idx = set(tree_.feature[tree_.feature >= 0])
used_features_kor = [feature_names_kor_list[i] for i in used_feature_idx]

st.subheader("📌 트리 요약")
st.write(f"- 답을 내는 마디(리프)는 모두 **{leaf_count}칸**이고, 그 중 **{no_stroke_leaf_count}칸**이 '아님'이라고 답합니다.")
if used_features_kor:
    st.write(f"- 고른 속성 가운데 이 나무가 실제로 물은 것은: **{', '.join(used_features_kor)}** 입니다.")
else:
    st.write("- 이 나무는 어떤 속성도 실제로 사용하지 않았습니다 (모든 사람에게 같은 답을 합니다).")
