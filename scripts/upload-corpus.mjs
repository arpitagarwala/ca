import {DatabaseSync} from 'node:sqlite';
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {spawn} from 'node:child_process';
import {sql} from './cloudflare.mjs';
const id='79a2cca9-859f-45a8-ac54-77fb7eaa76a0';
const db=new DatabaseSync('knowledge/cache/corpus.sqlite',{readOnly:true});
db.exec('PRAGMA busy_timeout=30000');
const partial=process.argv.includes('--partial');
const day=new Date().toISOString().slice(0,10);
let budget={day,writes:0};try{const b=JSON.parse(await readFile('knowledge/cache/upload-budget.json'));if(b.day===day)budget=b}catch{}
const signature=d=>createHash('sha256').update(d.sha256+'|'+JSON.stringify(JSON.parse(d.metadata).applicability)+'|'+d.title+'|'+d.type).digest('hex');
async function execute(query,params=[]){
 if(budget.writes>=85000)throw Error('Daily free ingestion write budget reached. Next daily run will resume.');
 for(let n=0;n<3;n++)try{
  const r=await sql(id,query,params);for(const result of r){if(result.success===false||result.error)throw Error(result.error||'D1 query unsuccessful');budget.writes+=result.meta?.rows_written||0}
  await writeFile('knowledge/cache/upload-budget.json',JSON.stringify(budget));return r;
 }catch(e){if(/quota|limit|budget/i.test(e.message)||n===2)throw e;await new Promise(r=>setTimeout(r,1000*(n+1)))}
}
const existing=(await execute('SELECT id,title,type,sha256,metadata,active FROM documents'))[0].results;
const remote=new Map(existing.map(d=>[d.id,d]));
const docs=db.prepare('SELECT * FROM documents WHERE active=1 ORDER BY level,subject_id,id').all();
let uploaded=0,unchanged=0;
const literal=v=>"'"+String(v).replaceAll("'","''")+"'";
if(process.argv.includes('--bulk')){
 const statements=[];let estimated=0;
 for(const doc of docs){
  const prior=remote.get(doc.id);if(prior&&signature(prior)===signature(doc)&&prior.active===1){unchanged++;continue}
  const chunks=db.prepare('SELECT page,page_end,title,text FROM chunks WHERE document_id=?').all(doc.id);
  estimated+=chunks.length*6+6;
  statements.push('DELETE FROM documents WHERE id='+literal(doc.id)+';');
  statements.push('INSERT INTO documents(id,level,subject_id,title,url,type,sha256,metadata,updated_at,active) VALUES('+[doc.id,doc.level,doc.subject_id,doc.title,doc.url,doc.type,doc.sha256,doc.metadata,doc.updated_at].map(literal).join(',')+',0);');
  for(const c of chunks)statements.push('INSERT INTO chunks(document_id,page,page_end,title,text) VALUES('+[doc.id,c.page,c.page_end,c.title,c.text].map(literal).join(',')+');');
  statements.push('UPDATE documents SET active=1 WHERE id='+literal(doc.id)+';');uploaded++;
 }
 if(budget.writes+estimated>85000)throw Error('Bulk import would exceed conservative daily free write budget. Run incremental sync to resume within budget.');
 if(statements.length){
  await writeFile('knowledge/cache/import.sql',statements.join('\n'));
  console.log(JSON.stringify({bulkDocuments:uploaded,estimatedWrites:estimated,unchanged}));
  await new Promise((resolve,reject)=>{
   const child=spawn(process.execPath,['node_modules/wrangler/bin/wrangler.js','d1','execute','ca-bhaiya-corpus','--remote','--file','knowledge/cache/import.sql','--yes'],{stdio:['ignore','pipe','pipe'],windowsHide:true});
   let output='';for(const stream of [child.stdout,child.stderr])stream.on('data',data=>{const text=data.toString();output+=text;process.stdout.write(text)});
   child.on('error',reject);child.on('exit',code=>code===0?resolve():reject(Error('Bulk import failed; completed document uploads can be resumed.')));
  });
  budget.writes+=estimated;await writeFile('knowledge/cache/upload-budget.json',JSON.stringify(budget));
 }
}else{
for(const doc of docs){
 const prior=remote.get(doc.id);
 if(prior&&signature(prior)===signature(doc)&&prior.active===1){unchanged++;continue}
 // Hide a document until all of its new chunks are successfully committed.
 await execute('DELETE FROM documents WHERE id=?',[doc.id]);
 await execute('INSERT INTO documents(id,level,subject_id,title,url,type,sha256,metadata,updated_at,active) VALUES(?,?,?,?,?,?,?,?,?,0)',[doc.id,doc.level,doc.subject_id,doc.title,doc.url,doc.type,doc.sha256,doc.metadata,doc.updated_at]);
 const chunks=db.prepare('SELECT page,page_end,title,text FROM chunks WHERE document_id=?').all(doc.id);
 for(let offset=0;offset<chunks.length;offset+=6){
  const batch=chunks.slice(offset,offset+6),values=batch.map(()=>'(?,?,?,?,?)').join(',');
  await execute('INSERT INTO chunks(document_id,page,page_end,title,text) VALUES '+values,batch.flatMap(c=>[doc.id,c.page,c.page_end,c.title,c.text]));
 }
 await execute('UPDATE documents SET active=1 WHERE id=?',[doc.id]);uploaded++;
 if(uploaded%10===0)console.log(JSON.stringify({uploaded,unchanged,examined:uploaded+unchanged,total:docs.length,writes:budget.writes}));
}
}
if(!partial){
 const active=new Set(docs.map(d=>d.id));
 for(const doc of existing)if(!active.has(doc.id)&&doc.active)await execute('UPDATE documents SET active=0 WHERE id=?',[doc.id]);
}
const active=(await execute('SELECT metadata FROM documents WHERE active=1'))[0].results.map(d=>JSON.parse(d.metadata));
let build={errors:[],pending:[]};try{build=JSON.parse(await readFile('knowledge/build-report.json','utf8'))}catch{}
const status={lastChecked:new Date().toISOString(),corpus:'d1-v5',revision:new Date().toISOString(),documents:active.map(({id,level,subjectId,title,url,type,applicability,pages,chunks,sha256})=>({id,level,subjectId,title,url,type,applicability,pages,chunks,sha256})),errors:build.errors,pending:build.pending,complete:!partial&&!build.errors.length,search:'indexed full-text; no embedding API',ingestionWritesToday:budget.writes};
await execute('INSERT INTO sync_state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',['status',JSON.stringify(status)]);
await writeFile('knowledge/sync-report.json',JSON.stringify(status,null,2));
console.log(JSON.stringify({uploaded,unchanged,active:active.length,writes:budget.writes,complete:status.complete}));
db.close();
