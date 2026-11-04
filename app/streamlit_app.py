# app/streamlit_app.py
"""GĐ6 — Ứng dụng Streamlit: dự đoán lương ĐĂNG TUYỂN cho vị trí IT tại Việt Nam.

Chạy từ thư mục gốc:  streamlit run app/streamlit_app.py
Cần có trước: models/salary_model.joblib, models/model_card.json (GĐ5: python src/evaluate.py).
Không có đăng nhập, database hay API backend: app chỉ đọc file trong dự án.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_cleaning import experience_group  # noqa: E402
from predict import InvalidInput, SalaryPredictor  # noqa: E402

DISCLAIMER = "Ước lượng từ tin tuyển dụng, không phải cam kết lương."
CLEAN_PATH = ROOT / "data" / "processed" / "it_salary_clean.csv"
SPLIT_PATH = ROOT / "data" / "processed" / "split.csv"
INTERIM_PATH = ROOT / "data" / "interim" / "vietjobs_it_candidates.csv"
RESULTS_PATH = ROOT / "reports" / "model_results.csv"
GROUP_MAE_PATH = ROOT / "reports" / "gd5_group_mae.csv"
PERM_PATH = ROOT / "reports" / "gd5_permutation_importance.csv"
TEST_PRED_PATH = ROOT / "reports" / "gd5_test_predictions.csv"
FIG_DIR = ROOT / "reports" / "figures"
RULES_PATH = ROOT / "reports" / "cleaning_rules.md"

# Nhãn hiển thị — giá trị bên trong giữ nguyên như mô hình dùng.
SENIORITY_LABELS = {
    "unknown": "Không ghi rõ", "intern": "Thực tập sinh", "fresher": "Fresher", "junior": "Junior",
    "middle": "Middle", "senior": "Senior", "lead": "Trưởng nhóm / Lead", "manager": "Quản lý / Manager",
}
# "Nhiều thành phố" là nhóm sinh ra khi làm sạch (tin ghi nhiều nơi), không phải lựa chọn của người dùng.
HIDDEN_CITIES = {"Nhiều thành phố"}
MIN_COMBO_N = 5  # tổ hợp (nhóm nghề × mức kinh nghiệm) có < 5 tin train -> cảnh báo ngoài phạm vi dữ liệu

st.set_page_config(page_title="Dự đoán lương IT", page_icon="💼", layout="centered")


# ----------------------------------------------------------------------------- tải dữ liệu/mô hình
@st.cache_resource
def load_predictor():
    return SalaryPredictor()


@st.cache_data
def load_csv(path: str):
    p = Path(path)
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


@st.cache_data
def load_train_frame():
    """Tập train (theo split.csv) — dùng để cảnh báo tổ hợp đầu vào hiếm. Không dùng test."""
    clean, split = load_csv(str(CLEAN_PATH)), load_csv(str(SPLIT_PATH))
    if clean is None or split is None:
        return None
    return clean.merge(split[["row_id", "split"]], on="row_id").query("split == 'train'")


def experience_label(y: float) -> str:
    if y == 0:
        return "Không yêu cầu"
    if y < 1:
        return f"{int(round(y * 12))} tháng"
    return f"{y:g} năm"


def experience_options(max_years: float) -> list:
    opts = [0.0, 0.25, 0.5] + [float(y) for y in range(1, int(max_years) + 1)]
    return [o for o in opts if o <= max_years]


try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()
card = predictor.card

page = st.sidebar.radio("Trang", ["Dự đoán", "Dữ liệu & Giới hạn", "Phân tích mô hình"])
st.sidebar.caption(DISCLAIMER)
st.sidebar.caption(f"Mô hình: {card['model']} · train {card['n_train']} tin · tạo {card['created_utc'][:10]}")


# ============================================================================= Trang 1: Dự đoán
def page_predict():
    st.title("Dự đoán lương đăng tuyển IT")
    st.write("Ước lượng **mức lương được ghi trong tin tuyển dụng** (trung điểm khoảng lương, "
             "triệu VND/tháng) cho vị trí IT tại Việt Nam. Đây không phải lương thực nhận.")

    allowed = card["allowed"]
    tcount = card["train_counts"]
    cities = [c for c in allowed["city"] if c not in HIDDEN_CITIES]
    exp_max = card["experience_years_range"][1]

    with st.form("predict_form"):
        job = st.selectbox("Nhóm nghề", allowed["job_category"],
                           format_func=lambda c: f"{c} ({tcount['job_category'].get(c, 0)} tin train)")
        city = st.selectbox("Thành phố", cities,
                            format_func=lambda c: f"{c} ({tcount['city'].get(c, 0)} tin train)")
        exp = st.select_slider("Kinh nghiệm yêu cầu", options=experience_options(exp_max), value=1.0,
                               format_func=experience_label,
                               help=f"Giới hạn 0–{exp_max:g} năm: đúng phạm vi dữ liệu huấn luyện.")
        seniority = st.selectbox("Cấp bậc ghi trong chức danh", allowed["seniority"],
                                 format_func=lambda s: SENIORITY_LABELS.get(s, s),
                                 help="Phần lớn tin không ghi cấp bậc; chọn 'Không ghi rõ' nếu không chắc.")
        skills = st.multiselect("Kỹ năng (tùy chọn)", card["skills_vocabulary"],
                                help="Chỉ gồm kỹ năng mô hình dùng (phổ biến nhất trong dữ liệu train).")
        submitted = st.form_submit_button("Dự đoán")

    perm = load_csv(str(PERM_PATH))
    if perm is not None and "skills" in set(perm["feature"]):
        v = float(perm.set_index("feature").loc["skills", "mae_increase"])
        if v < 0.2:
            st.caption(f"Lưu ý: trên tập test, kỹ năng hầu như không thay đổi độ chính xác của mô hình "
                       f"(permutation importance {v:+.2f} triệu). Kết quả phụ thuộc chủ yếu vào kinh nghiệm.")

    if not submitted:
        return
    try:
        r = predictor.predict(job_category=job, city=city, experience_years=exp,
                              seniority=seniority, skills=skills)
    except InvalidInput as e:
        st.error(f"Đầu vào không hợp lệ: {e}")
        return

    warnings = list(r["warnings"])
    train = load_train_frame()
    if train is not None:
        n_combo = int(((train["job_category"] == job) &
                       (train["experience_group"] == experience_group(exp))).sum())
        if n_combo < MIN_COMBO_N:
            warnings.append(f"Ngoài phạm vi dữ liệu: chỉ có {n_combo} tin train thuộc nhóm '{job}' với kinh "
                            f"nghiệm '{experience_label(exp)}'. Dự đoán dựa nhiều vào ngoại suy.")

    st.metric("Lương đăng tuyển ước lượng", f"{r['prediction']:.1f} {r['unit']}")
    lo, hi = r["range_80"]
    cov = card["residual_interval_oof"].get("test_coverage")
    st.write(f"Khoảng thường gặp (~80%): **{lo:.1f} – {hi:.1f} {r['unit']}**"
             + (f" — trên tập test, khoảng này chứa {100 * cov:.0f}% tin." if cov is not None else ""))
    cv = card.get("cv_metrics", {})
    tm = card.get("test_metrics", {})
    if cv and tm:
        st.caption(f"Sai số tuyệt đối trung bình (MAE): {cv['mae_mean']:.1f} ± {cv['mae_std']:.1f} triệu "
                   f"(cross-validation), {tm['mae_mean']:.1f} triệu (test, n={tm.get('n_test', '?')}).")
    for w in warnings:
        st.warning(w)
    st.info(DISCLAIMER)


# ============================================================================= Trang 2: Dữ liệu & Giới hạn
def page_data():
    st.title("Dữ liệu & Giới hạn")
    clean = load_csv(str(CLEAN_PATH))
    interim = load_csv(str(INTERIM_PATH))
    sha = card.get("data_sha256", {})

    st.subheader("Nguồn dữ liệu")
    st.write("- **VietJobs** (VinNLP, giấy phép MIT): 48.092 tin tuyển dụng đa ngành; theo README thu thập "
             "07–10/2025 (dataset không có ngày đăng nên chưa xác minh được).")
    if sha.get("vietjobs"):
        st.caption(f"sha256 file dùng: {sha['vietjobs']}")
    st.write("- **IT-Job-Posting** không có cột lương nên không dùng làm nhãn.")

    if interim is not None:
        st.subheader("Từ tin IT đến tập huấn luyện")
        it = interim[interim["is_it"]]
        funnel = it["model_reason"].value_counts().rename_axis("Kết quả").reset_index(name="Số tin")
        st.write(f"Có **{len(it):,}** tin được xếp là IT theo chức danh. Lý do giữ/loại:")
        st.dataframe(funnel, hide_index=True)

    if clean is not None:
        st.subheader(f"Tập dùng cho mô hình: {len(clean):,} tin")
        c1, c2 = st.columns(2)
        c1.write("Theo nhóm nghề")
        c1.dataframe(clean["job_category"].value_counts().rename_axis("Nhóm nghề").reset_index(name="Số tin"),
                     hide_index=True)
        c2.write("Theo thành phố")
        c2.dataframe(clean["city"].value_counts().rename_axis("Thành phố").reset_index(name="Số tin"),
                     hide_index=True)
        st.write("Phân bố lương đăng tuyển (trung điểm, triệu VND/tháng)")
        bins = pd.cut(clean["salary_label"], bins=[0, 5, 10, 15, 20, 25, 30, 40, 50, 100], right=False)
        counts = bins.value_counts().sort_index()  # sắp theo khoảng số, không theo chữ cái
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar([f"{iv.left:g}–{iv.right:g}" for iv in counts.index], counts.values, color="#4C72B0")
        ax.set(xlabel="triệu VND/tháng", ylabel="số tin")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        hn_share = 100 * (clean["city"] == "Hà Nội").mean()
        unk_share = 100 * (clean["seniority"] == "unknown").mean()
    else:
        hn_share = unk_share = None

    st.subheader("Giới hạn cần biết")
    lims = [
        "Nhãn là **trung điểm khoảng lương đăng tuyển** (ví dụ 15–25 triệu → 20 triệu): một giả định xấp xỉ, "
        "không phải lương thực nhận hay lương sau thương lượng.",
        "Tin ghi **'thỏa thuận'**, chỉ một con số, hoặc khoảng quá rộng (max/min > 5) bị loại; tin không phải "
        "toàn thời gian cũng bị loại.",
        "IT được xác định **theo chức danh** bằng quy tắc từ khóa (thiết kế đồ họa bị loại); quy tắc có thể "
        "bỏ sót hoặc bắt nhầm một số tin.",
        "Không có **ngày đăng** và **tên công ty**: không phân tích được xu hướng theo thời gian hay theo công ty.",
        "Mô hình chỉ biết các nhóm nghề, thành phố và kỹ năng có trong form; kinh nghiệm giới hạn "
        f"0–{card['experience_years_range'][1]:g} năm.",
        "Hệ số hay mức quan trọng của đặc trưng **không phải quan hệ nhân quả**.",
    ]
    if hn_share is not None:
        lims.insert(4, f"Dữ liệu lệch về **Hà Nội** ({hn_share:.0f}% số tin); {unk_share:.0f}% tin không ghi "
                       f"cấp bậc trong chức danh. Dự đoán cho thành phố/cấp bậc ít tin kém tin cậy hơn.")
    for x in lims:
        st.markdown(f"- {x}")
    if RULES_PATH.exists():
        with st.expander("Quy tắc làm sạch chi tiết"):
            st.markdown(RULES_PATH.read_text(encoding="utf-8"))
    st.info(DISCLAIMER)


# ============================================================================= Trang 3: Phân tích mô hình
def compare_test_cv(res: pd.DataFrame) -> str:
    """So sánh MAE test với MAE CV — mọi câu đều TÍNH từ số liệu, không viết cứng kết luận.
    Kiểm tra 2 điều độc lập: (1) mô hình chính test so với CV; (2) baseline test so với CV —
    nếu cả baseline cũng lệch quá 1 SD thì độ khó của tập test khác train."""
    def get(model, stage, col="mae_mean"):
        r = res[(res["model"] == model) & (res["stage"] == stage)]
        return float(r[col].iloc[0]) if len(r) and pd.notna(r[col].iloc[0]) else None

    m = card["model"]
    cv, cv_sd, te = get(m, "cv_train"), get(m, "cv_train", "mae_std"), get(m, "test")
    d_cv, d_sd, d_te = get("dummy_median", "cv_train"), get("dummy_median", "cv_train", "mae_std"), \
        get("dummy_median", "test")
    if None in (cv, cv_sd, te):
        return ""
    if abs(te - cv) <= cv_sd:
        msg = f"MAE test ({te:.2f}) nằm trong khoảng 1 SD của MAE cross-validation ({cv:.2f} ± {cv_sd:.2f})."
    else:
        msg = (f"MAE test ({te:.2f}) {'thấp hơn' if te < cv else 'cao hơn'} MAE cross-validation "
               f"({cv:.2f} ± {cv_sd:.2f}) quá 1 SD.")
    if None not in (d_cv, d_sd, d_te) and abs(d_te - d_cv) > d_sd:
        easier = d_te < d_cv
        msg += (f" Tuy vậy baseline cũng {'thấp hơn' if easier else 'cao hơn'} rõ trên test ({d_te:.2f} so với "
                f"{d_cv:.2f} ± {d_sd:.2f} CV), nên tập test có vẻ {'dễ' if easier else 'khó'} hơn train.")
        train = load_train_frame()
        clean, split = load_csv(str(CLEAN_PATH)), load_csv(str(SPLIT_PATH))
        if train is not None and clean is not None and split is not None:
            test_y = clean.merge(split[["row_id", "split"]], on="row_id").query("split == 'test'")["salary_label"]
            msg += (f" Độ lệch chuẩn của lương: train {train['salary_label'].std():.2f}, "
                    f"test {test_y.std():.2f} triệu.")
        msg += " Sai số kỳ vọng khi dùng thực tế nên lấy theo cross-validation."
    else:
        msg += " Baseline trên test cũng tương đương CV: kết quả nhất quán."
    return msg


def page_analysis():
    st.title("Phân tích mô hình")
    st.write(f"Mô hình dùng trong app: **{card['model']}** "
             f"(tham số: `{json.dumps(card['params'], ensure_ascii=False)}`). "
             "Được chọn ở GĐ4 bằng cross-validation theo nhóm trên train và quy tắc 1 độ lệch chuẩn "
             "(chọn mô hình đơn giản nhất trong khoảng 1 SD của mô hình tốt nhất). Tập test chỉ dùng một lần.")

    res = load_csv(str(RESULTS_PATH))
    if res is not None:
        st.subheader("So sánh mô hình (MAE, RMSE: triệu VND/tháng)")
        cols = [c for c in ["model", "stage", "mae_mean", "mae_std", "mae_ci_low", "mae_ci_high",
                            "rmse_mean", "r2_mean"] if c in res.columns]
        view = res[cols].sort_values(["stage", "mae_mean"])
        num = [c for c in cols if c not in ("model", "stage")]
        view = view.copy()
        for c in num:  # chuyển sẵn thành chuỗi: mọi phiên bản Streamlit đều hiện "—" thay vì "None"
            view[c] = view[c].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
        st.dataframe(view, hide_index=True)
        st.caption("stage = cv_train: GroupKFold 5 fold trên train (mae_std = độ lệch giữa các fold). "
                   "stage = test: đánh giá cuối, CI 95% bằng bootstrap theo nhóm.")

    preds = load_csv(str(TEST_PRED_PATH))
    if preds is not None:
        bias = float(preds["residual"].mean())
        st.write(f"Trên tập test ({len(preds)} tin), phần dư trung bình là **{bias:+.2f} triệu** "
                 f"({'mô hình dự đoán cao hơn thực tế' if bias < 0 else 'mô hình dự đoán thấp hơn thực tế'} "
                 "trung bình).")
    if res is not None:
        st.write(compare_test_cv(res))

    c1, c2 = FIG_DIR / "gd5_actual_vs_pred.png", FIG_DIR / "gd5_residuals.png"
    if c1.exists():
        st.subheader("Thực tế và dự đoán (test)")
        st.image(str(c1), width=460)
    if c2.exists():
        st.subheader("Phần dư (test)")
        st.image(str(c2))

    gm = load_csv(str(GROUP_MAE_PATH))
    if gm is not None:
        st.subheader("MAE theo nhóm trên test (kèm số mẫu)")
        names = {"experience_group": "Kinh nghiệm", "job_category": "Nhóm nghề", "city": "Thành phố",
                 "seniority_coarse": "Cấp bậc (gộp)"}
        col = st.selectbox("Nhóm theo", list(names), format_func=names.get)
        st.dataframe(gm[gm["column"] == col].drop(columns="column").round(2), hide_index=True)
        st.caption("Nhóm có n < 10 tin: chỉ để tham khảo, không đủ để kết luận.")

    perm = load_csv(str(PERM_PATH))
    if perm is not None:
        st.subheader("Mô hình dựa vào đầu vào nào? (permutation importance trên test)")
        names_vi = {"experience_years": "Kinh nghiệm", "seniority_coarse": "Cấp bậc", "job_category": "Nhóm nghề",
                    "city": "Thành phố", "skills": "Kỹ năng"}
        pv = perm.sort_values("mae_increase")
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh([names_vi.get(f, f) for f in pv["feature"]], pv["mae_increase"],
                xerr=pv["std"] if "std" in pv else None, color="#4C72B0")
        ax.axvline(0, color="gray", lw=0.8)
        ax.set(xlabel="MAE tăng thêm khi xáo trộn (triệu VND/tháng)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Mức tăng MAE (triệu) khi xáo trộn ngẫu nhiên một đầu vào. Cho biết mô hình dựa vào đầu vào "
                   "đó bao nhiêu để dự đoán — KHÔNG có nghĩa đầu vào đó gây ra mức lương.")
    st.info(DISCLAIMER)


{"Dự đoán": page_predict, "Dữ liệu & Giới hạn": page_data, "Phân tích mô hình": page_analysis}[page]()
