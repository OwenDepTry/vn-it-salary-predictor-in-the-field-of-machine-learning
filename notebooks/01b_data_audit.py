from pathlib import Path
import re
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data/raw/vietjobs/VietJobs.csv'
IT_CATEGORY = 'công_nghệ_thông_tin_kỹ_thuật_số'
IT_TITLE_PATTERN = re.compile(
    r'\b(?:it|cntt)\b|cong nghe thong tin|lap trinh|programmer|developer|'
    r'devops|software|tester|kiem thu (?:phan mem|game|ung dung|apps?)|'
    r'\bsap\b|data (?:analyst|engineer|scientist)|'
    r'system admin|sysadmin|network engineer|quan tri mang|an ninh mang|'
    r'cybersecurity|backend|front[\s-]?end|full[\s-]?stack|php|java|python',
)
GRAPHIC_DESIGN_PATTERN = re.compile(
    r'thiet ke|graphic designer|designer',
)
WEB_DESIGN_PATTERN = re.compile(
    r'\b(?:ui|ux)\b|web|website|php|frontend|front[\s-]?end',
)
FALSE_POSITIVE_PATTERN = re.compile(
    r'kiem thu kho|sap xep|video editor|digital marketing',
)
NUMERIC_RANGE_PATTERN = re.compile(
    r'(\d+(?:[.,]\d+)?)\s*[-–—]\s*(\d+(?:[.,]\d+)?)\s*trieu',
)
NUMERIC_SINGLE_PATTERN = re.compile(r'(\d+(?:[.,]\d+)?)\s*trieu')


def normalize_text(value):
    text = unicodedata.normalize('NFD', str(value).lower())
    return ''.join(char for char in text if unicodedata.category(char) != 'Mn')


def salary_class(value):
    text = normalize_text(value).strip()
    if text in ('', 'nan', 'none'):
        return 'missing'
    if re.search(r'\b(?:thoa thuan|thuong luong)\b', text):
        return 'negotiable'

    match = NUMERIC_RANGE_PATTERN.fullmatch(text)
    if match:
        lower = float(match.group(1).replace(',', '.'))
        upper = float(match.group(2).replace(',', '.'))
        if lower == 0:
            return 'range_lower_zero'
        if lower > upper:
            return 'range_reversed'
        return 'range'

    match = NUMERIC_SINGLE_PATTERN.fullmatch(text)
    if match:
        amount = float(match.group(1).replace(',', '.'))
        return 'negotiable' if amount == 0 else 'single'
    return 'other'


def title_review(title, category_match):
    normalized = normalize_text(title)
    false_positive = bool(FALSE_POSITIVE_PATTERN.search(normalized))
    graphic_design = bool(GRAPHIC_DESIGN_PATTERN.search(normalized))
    technical_design = bool(WEB_DESIGN_PATTERN.search(normalized))

    if false_positive:
        return 'likely_non_it', 'known_keyword_false_positive'
    if graphic_design and not technical_design:
        return 'likely_non_it', 'graphic_design_title'
    if category_match:
        return 'likely_it', 'it_category'
    if IT_TITLE_PATTERN.search(normalized):
        return 'manual_review', 'it_title_keyword'
    return 'manual_review', 'category_or_title_review'


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'{DATA_PATH} not found; run src/download_data.py first')

    df = pd.read_csv(DATA_PATH, low_memory=False)
    required = {
        'job_title', 'salary', 'salary_min', 'salary_max', 'category',
        'country', 'contract_type', 'location', 'description',
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')

    df = df.copy()
    df['salary_class'] = df['salary'].map(salary_class)
    df['salary_min_numeric'] = pd.to_numeric(df['salary_min'], errors='coerce')
    df['salary_max_numeric'] = pd.to_numeric(df['salary_max'], errors='coerce')

    country_mask = df['country'].fillna('').str.strip().eq('Việt Nam')
    vietnam = df.loc[country_mask].copy()
    category_match = vietnam['category'].fillna('').eq(IT_CATEGORY)
    normalized_titles = vietnam['job_title'].fillna('').map(normalize_text)
    title_match = normalized_titles.map(lambda title: bool(IT_TITLE_PATTERN.search(title)))
    title_false_positive = normalized_titles.map(
        lambda title: bool(FALSE_POSITIVE_PATTERN.search(title))
    )
    graphic_design = normalized_titles.map(
        lambda title: bool(GRAPHIC_DESIGN_PATTERN.search(title))
    )
    web_design = normalized_titles.map(
        lambda title: bool(WEB_DESIGN_PATTERN.search(title))
    )
    likely_non_it = title_false_positive | (graphic_design & ~web_design)

    # Keep title false positives in the review output, then exclude them from the funnel.
    it_candidate = category_match | title_match
    candidates = vietnam.loc[it_candidate].copy()
    candidate_category_match = category_match.loc[it_candidate]
    candidate_review = [
        title_review(title, is_category)
        for title, is_category in zip(candidates['job_title'], candidate_category_match)
    ]
    candidates['review_status'] = [status for status, _ in candidate_review]
    candidates['review_reason'] = [reason for _, reason in candidate_review]
    candidates['likely_non_it'] = likely_non_it.loc[it_candidate].to_numpy()
    eligible = candidates.loc[~candidates['likely_non_it']].copy()

    has_numeric_salary = eligible['salary_class'].isin(
        ['range', 'single', 'range_lower_zero', 'range_reversed']
    )
    valid_range = (
        eligible['salary_class'].eq('range')
        & eligible['salary_min_numeric'].gt(0)
        & eligible['salary_max_numeric'].ge(eligible['salary_min_numeric'])
    )
    full_time = eligible['contract_type'].fillna('').str.contains(
        'toàn thời gian', case=False, regex=False
    )
    within_salary_limits = (
        eligible['salary_min_numeric'].ge(3)
        & eligible['salary_max_numeric'].le(200)
    )
    deduplication_columns = ['job_title', 'location', 'salary', 'description']

    funnel = [
        ('all_vietjobs', len(df)),
        ('country_vietnam', int(country_mask.sum())),
        ('it_title_screened_candidates', len(eligible)),
        ('numeric_salary_stated', int(has_numeric_salary.sum())),
        ('valid_salary_range', int(valid_range.sum())),
        ('full_time_contract', int((valid_range & full_time).sum())),
        ('salary_range_3_to_200_million', int(
            (valid_range & full_time & within_salary_limits).sum()
        )),
        ('unique_title_location_salary_description', int(
            eligible.loc[valid_range & full_time & within_salary_limits]
            .drop_duplicates(subset=deduplication_columns)
            .shape[0]
        )),
    ]

    report_dir = ROOT / 'reports'
    report_dir.mkdir(exist_ok=True)
    review_columns = [
        'job_title', 'category', 'salary', 'salary_min', 'salary_max',
        'location', 'experience_required', 'salary_class', 'review_status',
        'review_reason', 'likely_non_it',
    ]
    review_columns = [column for column in review_columns if column in candidates]
    candidates[review_columns].to_csv(
        report_dir / 'gd1b_it_title_review.csv', index=False
    )
    pd.DataFrame(funnel, columns=['stage', 'rows']).to_csv(
        report_dir / 'gd1b_funnel.csv', index=False
    )

    final_count = funnel[-1][1]
    if final_count >= 800:
        decision = 'GO: đủ ngưỡng dữ liệu; tiếp tục với phạm vi đã loại chức danh không thuộc IT.'
    elif final_count >= 300:
        decision = 'GO có điều kiện, thu hẹp phạm vi: chỉ tiếp tục sau rà soát chức danh và luật làm sạch.'
    else:
        decision = 'NO-GO: số tin còn lại dưới 300.'

    salary_counts = df['salary_class'].value_counts(dropna=False)
    review_counts = candidates['review_status'].value_counts(dropna=False)
    excluded_title_count = int((likely_non_it & (category_match | title_match)).sum())
    summary = [
        'GD1 DATA AUDIT',
        f'Data: {DATA_PATH.relative_to(ROOT)}',
        f'rows={len(df)}',
        f'it_category_candidates_in_vietnam={int(category_match.sum())}',
        f'it_category_or_title_candidates_before_title_review={len(candidates)}',
        f'likely_non_it_candidates_excluded={excluded_title_count}',
        f'it_title_screened_candidates={len(eligible)}',
        'Salary string categories (all VietJobs):',
        *[f'  {name}: {count}' for name, count in salary_counts.items()],
        'Title review statuses (candidate set):',
        *[f'  {name}: {count}' for name, count in review_counts.items()],
        'GO/NO-GO thresholds set before decision: GO >= 800; conditional narrowed-scope GO 300-799; NO-GO < 300.',
        f'GO/NO-GO: {decision}',
        'Salary bounds are used only for the audit funnel, not as model features.',
        'IT-Job-Posting has no salary label.',
    ]
    text = '\n'.join(summary)
    (report_dir / 'gd1b_audit_output.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
