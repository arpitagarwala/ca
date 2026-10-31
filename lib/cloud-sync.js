import {verifyGitHubIdentity} from './github-identity.js';
import {officialURL,LEVELS} from './knowledge.js';
import {reserve} from './providers.js';
const reply=(body,status=200)=>Response.json(body,{status,headers:{'Cache-Control':'no-store'}});
const validId=id=>typeof id==='string'&&/^[a-f0-9]{32}$/.test(id);
const summary=doc=>{const{id,level,subjectId,title,url,type,applicability,pages,chunks,sha256}=doc;return{id,level,subjectId,title,url,type,applicability,pages,chunks,sha256}};
export async function handleCloudSync(request,env,options={}){
 let identity;try{identity=await (options.identityVerifier||verifyGitHubIdentity)(request.headers.get('Authorization'))}catch{return reply({error:'A valid identity from the approved GitHub workflow is required.'},401)}
 if(!env.CORPUS)return reply({error:'Corpus not configured'},503);
 if(request.method!=='POST')return reply({error:'Use POST'},405);
 const raw=await request.text();if(raw.length>8000000)return reply({error:'Payload too large'},413);
 let body;try{body=JSON.parse(raw)}catch{return reply({error:'Invalid JSON'},400)}
 const db=env.CORPUS,run=String(identity.run_id)+':'+String(identity.run_attempt),day=new Date().toISOString().slice(0,10);
 async function allowance(amount){if(!await reserve(db,'cloud-sync-writes:'+day,amount,40000))throw Error('Daily included write budget reached; next cloud run will resume')}
 try{
 if(body.operation==='snapshot'){
  const records=(await db.prepare('SELECT id,title,type,sha256,metadata,active FROM documents').all()).results;
  return reply({records});
 }
 if(body.operation==='begin'){
  if(!Array.isArray(body.ids)||body.ids.length<500||body.ids.length>5000||!body.ids.every(validId)||!Array.isArray(body.changed)||!body.changed.every(validId))return reply({error:'Invalid corpus membership'},400);
  // GitHub's concurrency group serializes jobs; a newer trusted run can resume an
  // interrupted attempt. Subsequent requests must match the newly recorded run.
  const removed=(await db.prepare('SELECT count(*) n FROM documents WHERE active=1 AND id NOT IN (SELECT value FROM json_each(?))').bind(JSON.stringify(body.ids)).first()).n;
  await allowance(body.changed.length*3+removed*3+5);
  const pending={run,started:Date.now(),ids:body.ids,changed:body.changed,staged:[]};
  await db.batch([
   db.prepare('UPDATE documents SET active=0 WHERE active=1 AND (id NOT IN (SELECT value FROM json_each(?)) OR id IN (SELECT value FROM json_each(?)))').bind(JSON.stringify(body.ids),JSON.stringify(body.changed)),
   db.prepare("INSERT INTO sync_state(key,value) VALUES('cloud-pending',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value").bind(JSON.stringify(pending))
  ]);
  return reply({started:true,expected:body.ids.length});
 }
 const row=await db.prepare("SELECT value FROM sync_state WHERE key='cloud-pending'").first();
 const pending=row?.value?JSON.parse(row.value):null;
 if(!pending||pending.run!==run)return reply({error:'Start a sync for this workflow run first'},409);
 if(body.operation==='put'){
  const d=body.document,parts=body.chunks;
  if(!d||!validId(d.id)||!pending.ids.includes(d.id)||!pending.changed.includes(d.id)||!LEVELS.includes(d.level)||!d.subjectId?.startsWith(d.level+'-')||d.approved!==true||!officialURL(d.url)||!officialURL(d.applicabilitySource)||!d.applicability||!['exact','from'].includes(d.applicability.kind)||!/^[a-f0-9]{64}$/.test(d.sha256||'')||!Array.isArray(parts)||!parts.length||parts.length>250)return reply({error:'Invalid official source record'},400);
  if(parts.some(c=>typeof c.text!=='string'||c.text.length<50||c.text.length>16000||!Number.isInteger(c.page)||c.page<1||!Number.isInteger(c.page_end)||c.page_end<c.page))return reply({error:'Invalid source passage'},400);
  const old=await db.prepare('SELECT count(*) n FROM chunks WHERE document_id=?').bind(d.id).first();
  await allowance((parts.length+(old?.n||0))*8+12);
  const statements=[db.prepare('DELETE FROM documents WHERE id=?').bind(d.id),db.prepare('INSERT INTO documents(id,level,subject_id,title,url,type,sha256,metadata,updated_at,active) VALUES(?,?,?,?,?,?,?,?,?,0)').bind(d.id,d.level,d.subjectId,d.title,d.url,d.type,d.sha256,JSON.stringify(d),new Date().toISOString())];
  for(let i=0;i<parts.length;i+=8){const group=parts.slice(i,i+8);statements.push(db.prepare('INSERT INTO chunks(document_id,page,page_end,title,text) VALUES '+group.map(()=>'(?,?,?,?,?)').join(',')).bind(...group.flatMap(c=>[d.id,c.page,c.page_end,d.title,c.text])))}
  pending.staged=[...new Set([...pending.staged,d.id])];
  statements.push(db.prepare("UPDATE sync_state SET value=? WHERE key='cloud-pending'").bind(JSON.stringify(pending)));
  await db.batch(statements);
  return reply({staged:d.id,passages:parts.length});
 }
 if(body.operation==='finish'){
  if(pending.changed.some(id=>!pending.staged.includes(id)))return reply({error:'Changed documents have not all been staged; activation refused'},409);
  const records=(await db.prepare('SELECT metadata FROM documents WHERE id IN (SELECT value FROM json_each(?))').bind(JSON.stringify(pending.ids)).all()).results.map(r=>JSON.parse(r.metadata));
  if(records.length!==pending.ids.length||LEVELS.some(l=>records.filter(d=>d.level===l).length<100))return reply({error:'Incomplete corpus; activation refused'},409);
  if(!body.catalog||LEVELS.some(l=>!Array.isArray(body.catalog.levels?.[l]?.subjects)||!body.catalog.levels[l].subjects.length))return reply({error:'Invalid course directory'},400);
  await allowance(records.length*3+10);
  const now=new Date().toISOString();
  const status={lastChecked:now,corpus:'d1-v5',revision:now,documents:records.map(summary),errors:body.errors||[],pending:body.pending||[],warnings:body.warnings||[],complete:!(body.errors||[]).length,search:'indexed full-text; no embedding API',updater:{provider:'github-actions',runId:identity.run_id,runUrl:'https://github.com/arpitagarwala/ca/actions/runs/'+identity.run_id,checkedAt:now}};
  await db.batch([
   db.prepare('UPDATE documents SET active=1 WHERE active=0 AND id IN (SELECT value FROM json_each(?))').bind(JSON.stringify(pending.ids)),
   db.prepare("INSERT INTO sync_state(key,value) VALUES('status',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value").bind(JSON.stringify(status)),
   db.prepare("INSERT INTO sync_state(key,value) VALUES('catalog',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value").bind(JSON.stringify(body.catalog)),
   db.prepare("DELETE FROM sync_state WHERE key='cloud-pending'")
  ]);
  return reply({complete:status.complete,documents:records.length,updater:status.updater});
 }
 return reply({error:'Unsupported sync operation'},400);
 }catch(error){return reply({error:/budget/.test(error.message)?error.message:'Cloud source sync failed; activation was not completed.'},/budget/.test(error.message)?429:503)}
}
