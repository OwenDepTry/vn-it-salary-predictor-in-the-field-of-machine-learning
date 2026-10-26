from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
 lines=[]
 for name in ['vietjobs.csv','it_job_postings.csv']:
  path=ROOT/'data/raw'/name
  if not path.exists():lines.append(f'{name}: MISSING (run download first)');continue
  df=pd.read_csv(path,low_memory=False)
  lines.extend([f'\n{name}: {len(df)} rows, {len(df.columns)} columns',f'columns: {list(df.columns)}',f'has salary: {"salary" in df.columns}'])
  for c in df.columns:lines.append(f'{c} | dtype={df[c].dtype} | missing={df[c].isna().sum()} ({df[c].isna().mean():.2%}) | unique={df[c].nunique()}')
 out='\n'.join(lines); (ROOT/'reports/gd1a_schema_output.txt').write_text(out,encoding='utf-8');print(out)
if __name__=='__main__':main()
