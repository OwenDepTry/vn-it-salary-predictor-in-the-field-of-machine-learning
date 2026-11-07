import hashlib, json, datetime, os, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'vietjobs': (
  'https://huggingface.co/datasets/dinhieufam/VietJobs/resolve/main/VietJobs.csv',
  Path('data/raw/vietjobs/VietJobs.csv'),
 ),
 'it_job_posting': (
  'https://raw.githubusercontent.com/SonPhatTranDeveloper/IT-Job-Posting/main/job_descriptions.csv',
  Path('data/raw/it_job_posting/job_descriptions.csv'),
 ),
}
def sha256_of(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def main():
 folder=ROOT/'data/raw'; folder.mkdir(parents=True,exist_ok=True)
 files=[]
 for name,(url,relative_path) in SOURCES.items():
  path=ROOT/relative_path
  path.parent.mkdir(parents=True,exist_ok=True)
  if not path.exists():
   print('Downloading',url)
   temporary_path=path.with_name(path.name+'.part')
   try:
    urllib.request.urlretrieve(url,temporary_path)
    os.replace(temporary_path,path)
   except Exception:
    temporary_path.unlink(missing_ok=True)
    raise
  files.append({
   'name':name,
   'url':url,
   'path':relative_path.as_posix(),
   'size_bytes':path.stat().st_size,
   'sha256':sha256_of(path),
  })
 log={
  'downloaded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'files':files,
 }
 (folder/'download_log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(log,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
