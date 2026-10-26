from pathlib import Path
import re, unicodedata
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
IT_CATEGORY='công_nghệ_thông_tin_kỹ_thuật_số'
def norm(x):
    s=unicodedata.normalize('NFD',str(x).lower())
    return ''.join(c for c in s if unicodedata.category(c)!='Mn')
def salary_class(x):
    s=norm(x).strip()
    if s in ('','nan','none'): return 'missing'
    if 'thoa thuan' in s or 'thuong luong' in s or re.search(r'\b0(?:[.,]0)?\s*trieu\b',s): return 'negotiable_or_zero'
    if re.search(r'\d+(?:[.,]\d+)?\s*[-–—]\s*\d+',s):return 'range'
    if re.search(r'\d',s):return 'single_or_other_numeric'
    return 'other'
def main():
    df=pd.read_csv(ROOT/'data/raw/vietjobs.csv',low_memory=False)
    required={'job_title','salary','salary_min','salary_max','category'}
    if not required.issubset(df):raise ValueError('Missing columns: '+str(required-set(df)))
    cat=df.category.fillna('').eq(IT_CATEGORY)
    lo=pd.to_numeric(df.salary_min,errors='coerce');hi=pd.to_numeric(df.salary_max,errors='coerce')
    numeric=(lo>0)&(hi>0)&(hi>=lo)
    classes=df.salary.map(salary_class)
    candidate=df.loc[cat].copy()
    candidate['salary_class']=classes[cat]
    candidate['numeric_range_consistent']=numeric[cat]
    candidate['review_status']='needs_manual_title_and_unit_review'
    (ROOT/'reports').mkdir(exist_ok=True)
    candidate[['job_title','category','salary','salary_min','salary_max','location','experience_required','salary_class','numeric_range_consistent','review_status']].to_csv(ROOT/'reports/gd1b_it_title_review.csv',index=False)
    steps=[('all_vietjobs',len(df)),('it_category_candidates',int(cat.sum())),('it_category_with_positive_consistent_salary_bounds',int((cat&numeric).sum()))]
    pd.DataFrame(steps,columns=['stage','rows']).to_csv(ROOT/'reports/gd1b_funnel.csv',index=False)
    summary=['GD1 DATA AUDIT — PRELIMINARY ONLY','Data: data/raw/vietjobs.csv',f'rows={len(df)}',f'it_category_candidates={int(cat.sum())}',f'positive_consistent_salary_bounds_all={int(numeric.sum())}',f'positive_consistent_salary_bounds_it={int((cat&numeric).sum())}', 'Salary string categories (all VietJobs):']
    summary += [f'  {k}: {v}' for k,v in classes.value_counts(dropna=False).items()]
    summary += ['Important: salary_min/max are numeric, salary_avg is a formatted text field.', 'GO/NO-GO: CONDITIONAL GO for continuing verification; NOT approved for training until TV2 verifies title, salary units/period, outliers, duplicates and clean target.', '449 clean records is a README claim, NOT a result produced by this script.','IT-Job-Posting has no salary label.']
    text='\n'.join(summary)
    (ROOT/'reports/gd1b_audit_output.txt').write_text(text,encoding='utf-8'); print(text)
if __name__=='__main__':main()
