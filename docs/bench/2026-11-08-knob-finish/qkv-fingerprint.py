import json,mmap,struct,sys
from pathlib import Path
p=Path(sys.argv[1])
with p.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
 offset=0
 def take(fmt):
  global offset
  r=struct.unpack_from('<'+fmt,data,offset);offset+=struct.calcsize('<'+fmt);return r[0] if len(r)==1 else r
 def string(read=True):
  global offset
  n=take('Q');start=offset;offset+=n
  return data[start:offset].decode() if read else None
 sizes={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
 formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
 def value(kind,read=False):
  global offset
  if kind==8:return string(read)
  if kind==9:
   typ=take('I');n=take('Q')
   if not read and typ in sizes:offset+=n*sizes[typ];return
   vals=[value(typ,read) for _ in range(n)]
   return vals if read else None
  if read:return take(formats[kind])
  offset+=sizes[kind]
 assert take('I')==0x46554747
 version=take('I');n_tensors=take('Q');n_kv=take('Q');metadata={}
 for _ in range(n_kv):
  name=string();typ=take('I');keep=not name.startswith('tokenizer.')
  v=value(typ,keep)
  if keep:metadata[name]=v
 tensors=[]
 for _ in range(n_tensors):
  name=string();nd=take('I');shape=[take('Q') for _ in range(nd)];dtype=take('I');pos=take('Q')
  if any(name.endswith(s) for s in ['attn_q.weight','attn_k.weight','attn_v.weight']):tensors.append({'name':name,'shape':shape,'ggml_type':dtype,'offset':pos})
 import hashlib,math
 alignment=metadata.get('general.alignment',32);base=(offset+alignment-1)//alignment*alignment
 for tensor in tensors:
  assert tensor['ggml_type']==8
  count=math.prod(tensor['shape']);assert count%32==0
  size=count//32*34;tensor['bytes']=size
  tensor['sha256']=hashlib.sha256(data[base+tensor['offset']:base+tensor['offset']+size]).hexdigest()
 print(json.dumps({'file':p.name,'file_bytes':len(data),'projection_bytes':sum(t['bytes'] for t in tensors),'attention_projections':tensors},indent=2))
