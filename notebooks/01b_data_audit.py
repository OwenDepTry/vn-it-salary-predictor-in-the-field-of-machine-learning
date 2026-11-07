"""GĐ1b — Kiểm toán dữ liệu VietJobs (TV1).

Đầu vào : data/raw/vietjobs/VietJobs.csv (không sửa)
Đầu ra  : reports/gd1b_audit_output.txt    (nhật ký kiểm toán + kết luận GO/NO-GO)
          reports/gd1b_funnel.csv          (phễu 8 bước x 3 cách lọc IT)
          reports/gd1b_it_title_review.csv (chức danh để rà soát bộ lọc IT)

Phễu GĐ1 chỉ để KHẢO SÁT: không loại tin chỉ vì nghi ngờ (vd. thiết kế đồ họa).
Các phát hiện được ghi lại để quyết định phạm vi; việc loại thật sự làm ở GĐ2 (TV2).
"""
from pathlib import Path
import re
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data/raw/vietjobs/VietJobs.csv'
REPORT_DIR = ROOT / 'reports'
OUT_TXT = REPORT_DIR / 'gd1b_audit_output.txt'
OUT_FUNNEL = REPORT_DIR / 'gd1b_funnel.csv'
OUT_REVIEW = REPORT_DIR / 'gd1b_it_title_review.csv'

IT_CATEGORY = 'công_nghệ_thông_tin_kỹ_thuật_số'
MIN_SALARY = 3.0     # triệu/tháng
MAX_SALARY = 200.0   # triệu/tháng
GO_THRESHOLD = 800
NARROW_THRESHOLD = 300

# ---------------------------------------------------------------- bộ lọc IT
# So khớp trên chức danh đã bỏ dấu + chữ thường.
IT_STRONG = re.compile('|'.join([
    r'\bdeveloper\b', r'\bdev\b', r'lap trinh', r'programmer', r'software', r'phan mem',
    r'back[\s-]?end', r'front[\s-]?end', r'full[\s-]?stack', r'devops', r'\bsre\b',
    r'data engineer', r'data scientist', r'machine learning', r'deep learning', r'\bml\b',
    r'\bios\b', r'android', r'flutter', r'react', r'\bjava\b', r'python', r'\.net\b', r'\bphp\b',
    r'golang', r'node\.?js', r'c\+\+', r'c#', r'tester', r'kiem thu', r'automation test',
    r'\bbrse\b', r'ky su cau noi',
    r'sysadmin|system admin|quan tri he thong|quan tri mang|network engineer|ky su mang',
    r'bao mat|security|cyber|pentest', r'\bdba\b|database|co so du lieu', r'\bcloud\b',
    r'\berp\b|\bsap\b|odoo', r'\bcntt\b|cong nghe thong tin', r'helpdesk|it support',
    r'embedded|lap trinh nhung|phan mem nhung', r'unity|game dev', r'blockchain',
    r'\bqa engineer\b|\bqc engineer\b|software qa',
]))
# "IT", "AI" so trên chức danh giữ nguyên chữ hoa để không bắt nhầm "ít", "ai".
IT_CASE_SENSITIVE = re.compile(r'\bIT\b|\bAI\b')
# Vị trí kinh doanh / tư vấn / marketing: không phải IT dù chức danh có từ khóa IT.
SALES_EXCLUDE = re.compile(r'kinh doanh|ban hang|sales|tu van|telesale|marketing|cham soc khach hang')
# Chỉ dùng để GHI NHẬN phát hiện, không dùng để loại trong phễu.
GRAPHIC_DESIGN = re.compile(r'thiet ke|designer|do hoa')
WEB_UI = re.compile(r'\bweb\b|website|\bui\b|\bux\b')
KNOWN_FALSE_POSITIVE = re.compile(r'kiem thu kho|sap xep|sap khai truong')

# ---------------------------------------------------------------- lương
RANGE_RE = re.compile(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*triệu')
SINGLE_RE = re.compile(r'(\d+(?:\.\d+)?)\s*triệu')


def strip_accents(value) -> str:
    """Bỏ dấu tiếng Việt (kể cả đ/Đ) + chữ thường, dùng để so khớp từ khóa."""
    text = '' if pd.isna(value) else str(value)
    text = text.replace('đ', 'd').replace('Đ', 'D')
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return text.lower()


def norm_text(value) -> str:
    """Chuẩn hóa để gộp trùng: chữ thường, gộp khoảng trắng."""
    return '' if pd.isna(value) else re.sub(r'\s+', ' ', str(value)).strip().lower()


def parse_salary(value):
    """Trả về (min, max, loại). Loại: range | range_lower_zero | range_reversed |
    single | negotiable | missing | other. "0.0 triệu" = thỏa thuận."""
    text = '' if pd.isna(value) else str(value).strip().lower()
    if text in ('', 'nan', 'none'):
        return None, None, 'missing'
    m = RANGE_RE.fullmatch(text)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo == 0:
            return lo, hi, 'range_lower_zero'
        if lo > hi:
            return lo, hi, 'range_reversed'
        return lo, hi, 'range'
    m = SINGLE_RE.fullmatch(text)
    if m:
        amount = float(m.group(1))
        return (None, None, 'negotiable') if amount == 0 else (amount, amount, 'single')
    return None, None, 'other'


LINES = []


def log(*parts):
    line = ' '.join(str(p) for p in parts)
    print(line)
    LINES.append(line)


def section(title):
    log('\n' + '=' * 80)
    log(title)
    log('=' * 80)


def top_titles(df, mask, n=20):
    return df.loc[mask, 'job_title'].value_counts().head(n).to_string()


def build_funnel(df, it_mask):
    steps = [('0. Tổng bản ghi VietJobs', len(df))]
    keep = pd.Series(True, index=df.index)

    def add(name, cond):
        nonlocal keep
        keep = keep & cond
        steps.append((name, int(keep.sum())))

    add('1. country == Việt Nam', df['country'].eq('Việt Nam'))
    add('2. thuộc IT', it_mask)
    add('3. có lương (bỏ 0.0 = thỏa thuận)', ~df['salary_class'].isin(['negotiable', 'missing', 'other']))
    add('4. dạng khoảng a–b hợp lệ (bỏ điểm đơn, 0–b, a>b)', df['salary_class'].eq('range'))
    add('5. hợp đồng Toàn thời gian', df['contract_type'].eq('Toàn thời gian'))
    add(f'6. {MIN_SALARY:g} ≤ min và max ≤ {MAX_SALARY:g} triệu',
        (df['sal_min'] >= MIN_SALARY) & (df['sal_max'] <= MAX_SALARY))
    subset = df.loc[keep]
    dup_index = subset.index[subset.duplicated(subset=['k_title', 'k_loc', 'salary', 'k_desc'], keep='first')]
    keep = keep.copy()
    keep.loc[dup_index] = False
    steps.append(('7. sau gộp trùng (tiêu đề+địa điểm+lương+mô tả)', int(keep.sum())))
    return steps, keep


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'{DATA_PATH} not found; run src/download_data.py first')
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    section('ĐỌC DỮ LIỆU')
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig', low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    log(f'VietJobs: {len(df):,} bản ghi, {df.shape[1]} cột')

    parsed = df['salary'].map(parse_salary)
    df['sal_min'] = [p[0] for p in parsed]
    df['sal_max'] = [p[1] for p in parsed]
    df['salary_class'] = [p[2] for p in parsed]

    title_ascii = df['job_title'].map(strip_accents)
    df['it_cat'] = df['category'].eq(IT_CATEGORY)
    df['it_title_kw'] = (title_ascii.str.contains(IT_STRONG)
                         | df['job_title'].astype(str).str.contains(IT_CASE_SENSITIVE))
    df['sales_title'] = title_ascii.str.contains(SALES_EXCLUDE)
    df['it_final'] = (df['it_cat'] | df['it_title_kw']) & ~df['sales_title']
    df['graphic_design'] = title_ascii.str.contains(GRAPHIC_DESIGN) & ~title_ascii.str.contains(WEB_UI)
    df['known_false_positive'] = title_ascii.str.contains(KNOWN_FALSE_POSITIVE)

    df['k_title'] = df['job_title'].map(norm_text)
    df['k_loc'] = df['location'].map(norm_text)
    df['k_desc'] = df['description'].map(norm_text)

    # ------------------------------------------------------------ 1. lương
    section('1. KIỂM TRA CỘT LƯƠNG (toàn bộ VietJobs)')
    counts = df['salary_class'].value_counts()
    log(counts.to_string())
    log(f'Tổng: {int(counts.sum()):,} (phải bằng số bản ghi)')
    is_tt = df['salary_avg'].map(strip_accents).str.contains('thoa thuan')
    log('\nĐối chiếu "0.0 triệu" với salary_avg == "Thoả thuận":')
    log(pd.crosstab(df['salary_class'].eq('negotiable'), is_tt,
                    rownames=['salary là 0.0'], colnames=['salary_avg thỏa thuận']).to_string())
    rng = df['salary_class'].eq('range')
    for col, p in [('salary_min', 'sal_min'), ('salary_max', 'sal_max')]:
        exact = (df.loc[rng, col] == df.loc[rng, p]).mean()
        floor_eq = (df.loc[rng, col] == df.loc[rng, p].astype(float).floordiv(1)).mean()
        log(f'{col} (cột số nguyên) so với giá trị parse từ chuỗi: bằng đúng={100 * exact:.1f}% | '
            f'bằng phần nguyên={100 * floor_eq:.1f}%')
    log('=> Dùng giá trị parse từ chuỗi `salary` để giữ phần thập phân.')

    # ------------------------------------------------------------ 2. bộ lọc IT
    section('2. BỘ LỌC IT')
    it_title = df['it_title_kw'] & ~df['sales_title']
    log(f'it_cat   (category == {IT_CATEGORY}): {int(df["it_cat"].sum()):,}')
    log(f'it_title (từ khóa chức danh, bỏ kinh doanh)       : {int(it_title.sum()):,}')
    log(f'bị loại vì chức danh kinh doanh/tư vấn/marketing  : '
        f'{int(((df["it_cat"] | df["it_title_kw"]) & df["sales_title"]).sum()):,}')
    log(f'it_final = (cat OR title) AND NOT kinh doanh      : {int(df["it_final"].sum()):,}')
    log('\nTop chức danh: category IT nhưng KHÔNG khớp từ khóa (category có lẫn nghề khác không?):')
    log(top_titles(df, df['it_cat'] & ~df['it_title_kw'] & ~df['sales_title']))
    log('\nTop chức danh: khớp từ khóa nhưng NGOÀI category IT (từ khóa có bắt nhầm không?):')
    log(top_titles(df, ~df['it_cat'] & df['it_title_kw'] & ~df['sales_title']))
    log('\nTop chức danh bị loại vì kinh doanh/tư vấn/marketing:')
    log(top_titles(df, (df['it_cat'] | df['it_title_kw']) & df['sales_title']))

    review = (df.loc[df['it_cat'] | df['it_title_kw']]
              .groupby('job_title')
              .agg(n=('job_title', 'size'),
                   category_top=('category', lambda s: s.value_counts().index[0]),
                   it_cat=('it_cat', 'max'), it_title_kw=('it_title_kw', 'max'),
                   sales_title=('sales_title', 'max'), it_final=('it_final', 'max'),
                   graphic_design=('graphic_design', 'max'),
                   known_false_positive=('known_false_positive', 'max'))
              .sort_values('n', ascending=False))
    review.to_csv(OUT_REVIEW, encoding='utf-8-sig')
    log(f'\nĐã lưu {len(review):,} chức danh để rà soát: reports/{OUT_REVIEW.name}')

    # ------------------------------------------------------------ 3. phễu
    section('3. PHỄU DỮ LIỆU')
    variants = {'it_cat': df['it_cat'], 'it_title': it_title, 'it_final': df['it_final']}
    tables, masks = {}, {}
    for name, mask in variants.items():
        steps, final_mask = build_funnel(df, mask)
        tables[name] = pd.Series({s: n for s, n in steps})
        masks[name] = final_mask
    funnel_df = pd.DataFrame(tables)
    log(funnel_df.to_string())
    funnel_df.to_csv(OUT_FUNNEL, encoding='utf-8-sig')
    log(f'Đã lưu: reports/{OUT_FUNNEL.name}')

    # ------------------------------------------------------------ 4. phát hiện khi rà soát
    section('4. PHÁT HIỆN KHI RÀ SOÁT (it_final, cuối phễu) — chỉ ghi nhận, không loại ở GĐ1')
    final = df.loc[masks['it_final']]
    log(f'Số tin cuối phễu: {len(final):,}')
    log('\nChức danh phổ biến nhất:')
    log(final['job_title'].value_counts().head(15).to_string())
    skills = (final['technical_skills'].fillna('').str.strip('[]').str.replace("'", '')
              .str.split(',').explode().str.strip().str.lower())
    log('\nKỹ năng phổ biến nhất:')
    log(skills[skills.ne('')].value_counts().head(10).to_string())
    log(f'\nChức danh thiết kế đồ họa (không phải web/UI): {int(final["graphic_design"].sum()):,} tin')
    log(final.loc[final['graphic_design'], 'job_title'].value_counts().head(10).to_string())
    log(f'\nTừ khóa bắt nhầm đã biết ("kiêm thủ kho" -> "kiểm thử", "Sắp xếp" -> "SAP"): '
        f'{int(final["known_false_positive"].sum()):,} tin')
    log(final.loc[final['known_false_positive'], 'job_title'].value_counts().head(10).to_string())

    # ------------------------------------------------------------ 5. GO/NO-GO
    section('5. GO/NO-GO')
    n_final = int(funnel_df['it_final'].iloc[-1])
    log(f'Ngưỡng đặt trước khi đếm: GO nếu ≥ {GO_THRESHOLD}; GO thu hẹp nếu {NARROW_THRESHOLD}–{GO_THRESHOLD}; '
        f'NO-GO nếu < {NARROW_THRESHOLD}.')
    log(f'Bộ lọc ban đầu (category CNTT hoặc từ khóa chức danh, bỏ kinh doanh) cho {n_final:,} tin.')
    log('Khi rà soát: category CNTT của VietJobs gộp cả thiết kế đồ họa (xem mục 4: chức danh "Thiết kế", '
        'kỹ năng photoshop), và bộ lọc từ khóa bắt nhầm một số tin do bỏ dấu tiếng Việt '
        '("kiêm thủ kho" thành "kiểm thử", "Sắp xếp" thành "SAP").')
    if n_final < NARROW_THRESHOLD:
        decision = 'NO-GO: dưới ngưỡng tối thiểu.'
    else:
        decision = ('GO có điều kiện, thu hẹp phạm vi: chỉ giữ IT theo nghĩa phần mềm/hạ tầng, loại thiết kế đồ họa '
                    'và lọc theo chức danh thay vì theo category (thực hiện ở GĐ2). Thiết kế đồ họa có mức lương và '
                    'kỹ năng khác hẳn, gộp chung sẽ lệch khỏi tên đề tài và làm nhiễu mô hình.')
    log(f'Quyết định: {decision}')
    log('IT-Job-Posting không có cột lương, chỉ dùng để khảo sát.')

    OUT_TXT.write_text('\n'.join(LINES), encoding='utf-8')


if __name__ == '__main__':
    main()