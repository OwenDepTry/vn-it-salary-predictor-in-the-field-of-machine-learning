from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'vietjobs': ROOT/'data/raw/vietjobs/VietJobs.csv',
 'it_job_posting': ROOT/'data/raw/it_job_posting/job_descriptions.csv',
}
def main():
 lines=[]
 for name,path in SOURCES.items():
  if not path.exists():raise FileNotFoundError(f'{path} not found; run src/download_data.py first')
  df=pd.read_csv(path,low_memory=False)
  lines.extend([f'\n{name}: {len(df)} rows, {len(df.columns)} columns',f'columns: {list(df.columns)}',f'has salary: {"salary" in df.columns}'])
  for c in df.columns:lines.append(f'{c} | dtype={df[c].dtype} | missing={df[c].isna().sum()} ({df[c].isna().mean():.2%}) | unique={df[c].nunique()}')
 out='\n'.join(lines); (ROOT/'reports/gd1a_schema_output.txt').write_text(out,encoding='utf-8');print(out)
if __name__=='__main__':main()
