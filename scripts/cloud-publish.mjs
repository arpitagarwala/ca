import {DatabaseSync} from 'node:sqlite';
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {SYNC_AUDIENCE} from '../lib/github-identity.js';
const endpoint=SYNC_AUDIENCE;
let bearer,expires=0;
async function identity(){
 if(bearer&&Date.now()<expires-60000)return bearer;
 if(!process.env.ACTIONS_ID_TOKEN_REQUEST_URL||!process.env.ACTIONS_ID_TOKEN_REQUEST_TOKEN)throw Error('Run this publisher only in the approved GitHub Actions workflow.');
 const url=new URL(process.env.ACTIONS_ID_TOKEN_REQUEST_URL);url.searchParams.set('audience',SYNC_AUDIENCE);
 const r=await fetch(url,{headers:{Authorization:'Bearer '+process.env.ACTIONS_ID_TOKEN_REQUEST_TOKEN},signal:AbortSignal.timeout(30000)});
 if(!r.ok)throw Error('Could not obtain GitHub workload identity');
 bearer=(await r.json()).value;
 expires=JSON.parse(Buffer.from(bearer.split('.')[1],'base64url').toString()).exp*1000;
 return bearer;
}
async function api(body){
 for(let n=0;n<3;n++){
  const r=await fetch(endpoint,{method:'POST',headers:{Authorization:'Bearer '+await identity(),'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(90000)});
  let value;try{value=await r.json()}catch{value={error:'Invalid cloud publisher response'}}
  if(r.ok)return value;
  if(r.status===401){bearer=null;expires=0}
  if(r.status===429||r.status===400||n===2)throw Error(value.error||'Cloud publication failed: '+r.status);
  await new Promise(resolve=>setTimeout(resolve,2000*(n+1)));
 }
}
const signature=d=>createHash('sha256').update(d.sha256+'|'+JSON.stringify(JSON.parse(d.metadata).applicability)+'|'+d.title+'|'+d.type).digest('hex');
const db=new DatabaseSync('knowledge/cache/corpus.sqlite',{readOnly:true});
const documents=db.prepare('SELECT * FROM documents WHERE active=1 ORDER BY level,subject_id,id').all();
const snapshot=await api({operation:'snapshot'}),remote=new Map(snapshot.records.map(d=>[d.id,d]));
// Matching staged records can be reused after a quota-limited or interrupted run.
const changed=documents.filter(d=>!remote.has(d.id)||signature(remote.get(d.id))!==signature(d));
await api({operation:'begin',ids:documents.map(d=>d.id),changed:changed.map(d=>d.id)});
console.log(JSON.stringify({verifiedRecords:documents.length,changed:changed.length,unchanged:documents.length-changed.length}));
let uploaded=0;
for(const doc of changed){
 const metadata=JSON.parse(doc.metadata),chunks=db.prepare('SELECT page,page_end,text FROM chunks WHERE document_id=?').all(doc.id);
 await api({operation:'put',document:metadata,chunks});uploaded++;
 if(uploaded%10===0)console.log(JSON.stringify({uploaded,total:changed.length}));
}
const catalog=JSON.parse(await readFile('public/catalog.json','utf8')),report=JSON.parse(await readFile('knowledge/build-report.json','utf8'));
const result=await api({operation:'finish',catalog,errors:report.errors,pending:report.pending});
await writeFile('knowledge/cloud-sync-status.json',JSON.stringify({checkedAt:new Date().toISOString(),complete:result.complete,records:documents.length,changed:changed.length,uploaded,failures:report.errors.length,runUrl:result.updater.runUrl},null,2)+'\n');
console.log(JSON.stringify(result));db.close();
