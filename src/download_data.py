import argparse, hashlib, json, datetime, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'vietjobs.csv': 'https://huggingface.co/datasets/dinhieufam/VietJobs/resolve/main/VietJobs.csv',
 'it_job_postings.csv': 'https://raw.githubusercontent.com/SonPhatTranDeveloper/IT-Job-Posting/main/job_descriptions.csv'
}
def sha256_of(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def main():
 folder=ROOT/'data/raw'; folder.mkdir(parents=True,exist_ok=True)
 log={}
 for name,url in SOURCES.items():
  path=folder/name
  if not path.exists():
   print('Downloading',url)
   try: urllib.request.urlretrieve(url,path)
   except Exception as e: print('FAILED',name,e);continue
  log[name]={'source_url':url,'sha256':sha256_of(path),'bytes':path.stat().st_size,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (folder/'download_log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(log,indent=2))
if __name__=='__main__':main()
