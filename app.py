import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="영화 흥행 예측기", layout="wide")
st.title("🎬 영화 흥행 예측기")
st.caption("KOBIS 데이터를 이용해 총 관객 수를 예측하는 다중 회귀 모델입니다.")

# -----------------------------
# 데이터 불러오기
# -----------------------------
DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    return daily, movies

daily_df, movies_df = load_data()

# -----------------------------
# 기간 계산 (일별 표 기준)
# -----------------------------
date_col = daily_df.columns[0]  # 날짜(여덟 자리 숫자) 열
dates = pd.to_datetime(daily_df[date_col].astype(str), format="%Y%m%d")
start_date = dates.min().strftime("%Y-%m-%d")
end_date = dates.max().strftime("%Y-%m-%d")

st.subheader("📅 데이터 기준 기간")
st.write(f"기준 기간: **{start_date} ~ {end_date}**")

# -----------------------------
# 영화별 표 원본 보여주기
# -----------------------------
st.subheader("📋 영화별 정보 표 (원본, 상위 5개 행)")
st.dataframe(movies_df.head())

# -----------------------------
# 영화코드 순 정렬 + train/test 분리
# 열 편마다 앞의 세 편을 테스트용으로 분리
# -----------------------------
movies_sorted = movies_df.sort_values("movieCd").reset_index(drop=True)

def split_train_test(df, group_size=10, test_size=3):
    test_idx = []
    train_idx = []
    for i in range(0, len(df), group_size):
        block = list(range(i, min(i + group_size, len(df))))
        test_idx.extend(block[:test_size])
        train_idx.extend(block[test_size:])
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

train_df, test_df = split_train_test(movies_sorted, group_size=10, test_size=3)

# -----------------------------
# ⚠️ 사후 집계값 안내
# -----------------------------
st.warning(
    "⚠️ **주의**: 이 표에 쓰인 `first_scrn`, `first_show`, `first_week_audi`, "
    "`days_in_top10` 같은 변수는 영화가 개봉하고 **어느 정도 시간이 지난 뒤에야 알 수 있는 사후 집계값**입니다. "
    "따라서 이 앱의 예측 점수는 '개봉 전에 흥행을 미리 맞히는 성능'이 아니라, "
    "'개봉 후 데이터를 안다면 총 관객 수와 얼마나 관련이 있는지'를 보여주는 것에 가깝습니다."
)

# -----------------------------
# 사용할 변수 선택 (체크박스) - 자유 탐색용
# -----------------------------
st.subheader("🧮 (자유 탐색) 학습에 사용할 변수 선택")

candidate_features = {
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "first_date": "10위권 첫 등장일",
    "peak": "성수기 개봉 여부(1/0)",
    "first_week_audi": "첫 주 관객",
    "days_in_top10": "10위권 유지 일수",
}

selected_features = []
cols = st.columns(3)
for i, (col_name, label) in enumerate(candidate_features.items()):
    with cols[i % 3]:
        if st.checkbox(f"{label} ({col_name})", value=True, key=col_name):
            selected_features.append(col_name)

if len(selected_features) == 0:
    st.warning("최소 1개 이상의 변수를 선택해주세요.")
    st.stop()

target = "total_audi"

# -----------------------------
# 공용 학습/평가 함수
# -----------------------------
def train_and_evaluate(features, train_df, test_df, target):
    use_cols = features + [target, "movieNm", "movieCd"]
    train_clean = train_df[use_cols].dropna()
    test_clean = test_df[use_cols].dropna()

    X_train = train_clean[features]
    y_train = train_clean[target]
    X_test = test_clean[features]
    y_test = test_clean[target]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, a_min=1, a_max=None)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "features": features,
        "train_clean": train_clean,
        "test_clean": test_clean,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "r2": r2,
    }

# -----------------------------
# 모델 A: 기본 변수 3가지
# 모델 B: 기본 변수 3가지 + 첫 주 관객
# -----------------------------
BASE_FEATURES = ["first_scrn", "first_show", "peak"]
EXTRA_FEATURES = BASE_FEATURES + ["first_week_audi"]

result_base = train_and_evaluate(BASE_FEATURES, train_df, test_df, target)
result_extra = train_and_evaluate(EXTRA_FEATURES, train_df, test_df, target)

# -----------------------------
# 두 모델 비교 표시
# -----------------------------
st.subheader("🆚 기본 변수 모델 vs 첫 주 관객 추가 모델 비교")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 📦 기본 변수 3가지")
    st.caption(f"사용 변수: {', '.join(BASE_FEATURES)}")
    st.metric("학습 영화 수", f"{len(result_base['train_clean'])} 편")
    st.metric("평가 영화 수", f"{len(result_base['test_clean'])} 편")
    st.metric("R²", f"{result_base['r2']:.3f}")
    st.metric("MAE", f"{result_base['mae']:,.0f} 명")

with col_b:
    st.markdown("### ➕ 기본 변수 + 첫 주 관객")
    st.caption(f"사용 변수: {', '.join(EXTRA_FEATURES)}")
    st.metric("학습 영화 수", f"{len(result_extra['train_clean'])} 편")
    st.metric("평가 영화 수", f"{len(result_extra['test_clean'])} 편")
    st.metric("R²", f"{result_extra['r2']:.3f}")
    st.metric("MAE", f"{result_extra['mae']:,.0f} 명")

st.caption(
    "첫 주 관객 수는 개봉 후 일주일이 지나야 알 수 있는 값이므로, "
    "두 모델의 점수 차이는 '개봉 직후 정보를 추가로 알면 총 관객 수와의 관련성이 얼마나 더 좋아지는지'를 보여줍니다. "
    "실제 개봉 전 예측과는 다르다는 점을 기억해주세요."
)

st.write(f"📅 기준 기간: **{start_date} ~ {end_date}**")

# -----------------------------
# 자유 탐색 모델 학습 (체크박스로 고른 변수)
# -----------------------------
result_custom = train_and_evaluate(selected_features, train_df, test_df, target)

y_test = result_custom["y_test"]
y_pred = result_custom["y_pred"]
test_clean = result_custom["test_clean"]
train_clean = result_custom["train_clean"]
model = result_custom["model"]
mae = result_custom["mae"]
r2 = result_custom["r2"]

# -----------------------------
# 결과 요약 (자유 탐색 모델)
# -----------------------------
st.subheader("📊 자유 탐색 모델 학습 및 평가 결과")

col1, col2, col3 = st.columns(3)
col1.metric("학습에 사용한 영화 수", f"{len(train_clean)} 편")
col2.metric("평가에 사용한 영화 수", f"{len(test_clean)} 편")
col3.metric("기준 기간", f"{start_date} ~ {end_date}")

col4, col5 = st.columns(2)
col4.metric("결정계수 (R²)", f"{r2:.3f}")
col5.metric("평균 절대 오차 (MAE)", f"{mae:,.0f} 명")

st.caption("R²는 1에 가까울수록 예측이 실제값의 패턴을 잘 설명한다는 뜻이고, "
           "MAE는 예측이 평균적으로 실제값과 몇 명 정도 차이 나는지를 나타냅니다.")

# -----------------------------
# 1,000명 미만 예측 처리
# -----------------------------
low_pred_count = int((y_pred < 1000).sum())
plot_y_pred = np.where(y_pred < 1000, 1000, y_pred)  # 바닥에 붙이기 위한 값

st.write(f"⚠️ 예측 관객 수가 1,000명보다 작게 나온 영화: **{low_pred_count}편** "
         f"(그래프에서는 바닥선에 붙여 표시됩니다.)")

# -----------------------------
# Plotly 산점도
# -----------------------------
st.subheader("📈 실제 관객 수 vs 예측 관객 수 (로그 스케일)")

fig = go.Figure()

# 대각선 (y = x)
min_val = max(1, min(y_test.min(), plot_y_pred.min()))
max_val = max(y_test.max(), plot_y_pred.max())
diag_range = [min_val, max_val]

fig.add_trace(go.Scatter(
    x=diag_range,
    y=diag_range,
    mode="lines",
    name="실제 = 예측",
    line=dict(color="gray", dash="dash")
))

fig.add_trace(go.Scatter(
    x=y_test,
    y=plot_y_pred,
    mode="markers",
    name="테스트 영화",
    text=test_clean["movieNm"],
    hovertemplate="영화명: %{text}<br>실제: %{x:,.0f}명<br>예측: %{y:,.0f}명<extra></extra>",
    marker=dict(size=9, color="royalblue", opacity=0.7)
))

fig.update_xaxes(type="log", title="실제 총 관객 수 (로그 스케일)")
fig.update_yaxes(type="log", title="예측 총 관객 수 (로그 스케일)")
fig.update_layout(
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 사용된 변수와 회귀 계수 참고 정보
# -----------------------------
with st.expander("🔍 회귀 계수 자세히 보기 (자유 탐색 모델)"):
    coef_df = pd.DataFrame({
        "변수": selected_features,
        "계수": model.coef_
    })
    st.dataframe(coef_df)
    st.write(f"절편(intercept): {model.intercept_:,.2f}")

with st.expander("🎞️ 테스트 영화 목록과 예측 결과 (자유 탐색 모델)"):
    result_df = test_clean[["movieNm", target]].copy()
    result_df["예측값"] = y_pred
    result_df.columns = ["영화명", "실제 관객수", "예측 관객수"]
    st.dataframe(result_df.sort_values("실제 관객수", ascending=False))
