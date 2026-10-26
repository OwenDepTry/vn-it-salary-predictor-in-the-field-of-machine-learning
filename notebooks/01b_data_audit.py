from pathlib import Path
import pandas as pd, re, unicodedata
ROOT=Path(__file__).resolve().parents[1]
def norm(s):
 s=unicodedata.normalize('NFD',str(s).lower());return ''.join(c for c in s if unicodedata.category(c)!='Mn')
def salary_type(s):
 s=norm(s).strip()
 if not s or s=='nan':return 'missing'
 if 'thoa thuan' in s or 'thuong luong' in s or re.search(r'\b0(?:[.,]0)?\s*trieu\b',s):return 'negotiable_or_zero'
 if re.search(r'\d+(?:[.,]\d+)?\s*[-–—]\s*\d+',s):return 'range'
 if re.search(r'\d',s):return 'single_or_other_numeric'
 return 'other'
def it_title(s):
 t=norm(s)
 return bool(re.search(r'\b(it|software|developer|programmer|backend|frontend|front.end|full.stack|devops|data engineer|data scientist|qa|tester|kiem thu phan mem|lap trinh|phat trien phan mem|an ninh mang|cybersecurity|quan tri mang|system administrator|tri tue nhan tao|machine learning)\b',t))
def main():
 df=pd.read_csv(ROOT/'data/raw/vietjobs.csv',low_memory=False)
 if not {'job_title','salary'}.issubset(df.columns):raise ValueError('Unexpected schema: '+str(df.columns.tolist()))
 types=df.salary.map(salary_type);mask=df.job_title.map(it_title)
 review=df.loc[mask,['job_title','salary','location','experience_required']].copy();review['salary_class']=types[mask];review['review_status']='needs_manual_review'
 review.to_csv(ROOT/'reports/gd1b_it_title_review.csv',index=False)
 steps=[('all_records',len(df)),('it_title_keyword_candidates',int(mask.sum())),('it_candidates_with_numeric_salary',int((mask & types.isin(['range','single_or_other_numeric'])).sum()))]
 pd.DataFrame(steps,columns=['stage','rows']).to_csv(ROOT/'reports/gd1b_funnel.csv',index=False)
 out=['AUDIT: candidate screening only; NOT a final clean dataset',f'Rows: {len(df)}',f'IT keyword candidates: {mask.sum()}','Salary classes:']+[f'{k}: {v}' for k,v in types.value_counts().items()]+['GO/NO-GO: pending manual title review and salary normalization by TV2.','Do NOT interpret 0.0 triệu as a zero salary.']
 txt='\n'.join(out);(ROOT/'reports/gd1b_audit_output.txt').write_text(txt,encoding='utf-8');print(txt)
if __name__=='__main__':main()
