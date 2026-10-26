from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 path=ROOT/'data/processed/it_salary_clean.csv'
 if not path.exists():
  print('WAITING FOR TV2: data/processed/it_salary_clean.csv');return
 df=pd.read_csv(path)
 candidates=['salary_target','salary_million','salary_mid','salary_avg','salary']
 target=next((c for c in candidates if c in df and pd.to_numeric(df[c],errors='coerce').notna().sum()>0),None)
 if target is None:raise ValueError('No numeric salary target; coordinate schema with TV2')
 y=pd.to_numeric(df[target],errors='coerce');lines=[f'rows={len(df)}',f'target={target}',str(y.describe())]
 figures=ROOT/'reports/figures';figures.mkdir(exist_ok=True)
 def save(name):plt.tight_layout();plt.savefig(figures/f'gd2_{name}.png',dpi=160);plt.close()
 plt.figure();y.dropna().hist(bins=30);plt.xlabel('Salary (dataset unit; verify with TV2)');plt.ylabel('Count');save('01_salary_hist')
 plt.figure();y.dropna().plot.box();plt.ylabel('Salary');save('02_salary_box')
 dimensions=[('seniority','03_seniority'),('city','04_city'),('job_group','05_job_group'),('experience_group','06_experience'),('skills','07_skills')]
 for col,name in dimensions:
  plt.figure(figsize=(10,5))
  if col in df:
   if col=='skills':df[col].fillna('').astype(str).value_counts().head(12).plot.bar()
   else:df.assign(_target=y).groupby(col)['_target'].median().sort_values(ascending=False).head(12).plot.bar()
  else:
   plt.text(.5,.5,f'Missing column: {col}\nAwait TV2 schema',ha='center',va='center');plt.axis('off');lines.append(f'WARNING missing {col}')
  plt.xticks(rotation=35,ha='right');save(name)
 (ROOT/'reports/gd2_eda_output.txt').write_text('\n'.join(lines),encoding='utf-8')
 print('\n'.join(lines))
if __name__=='__main__':main()
