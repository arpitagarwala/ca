import {test} from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync} from 'node:fs';
import {handleCloudSync} from '../lib/cloud-sync.js';
import {NOTICE_URLS} from '../lib/notice-fetch.js';
const identityVerifier=async()=>({run_id:'test-run',run_attempt:'1'});
function database(){
 const native=new DatabaseSync(':memory:');native.exec(readFileSync(new URL('../knowledge/schema.sql',import.meta.url),'utf8'));
 const ids=[];
 for(let n=0;n<500;n++){
  const id=n.toString(16).padStart(32,'0'),level=['foundation','intermediate','final'][n%3];ids.push(id);
  const meta={id,level,subjectId:level+'-1',title:'Official chapter',url:'https://resource.cdn.icai.org/chapter.pdf',type:'study_material',sha256:'a'.repeat(64),approved:true,applicability:{kind:'exact',attempts:['May 2026']},applicabilitySource:'https://www.icai.org/post/study-material-nset',pages:1,chunks:1};
  native.prepare('INSERT INTO documents(id,level,subject_id,title,url,type,sha256,metadata,updated_at) VALUES(?,?,?,?,?,?,?,?,?)').run(id,level,meta.subjectId,meta.title,meta.url,meta.type,meta.sha256,JSON.stringify(meta),'today');
 }
 const db={prepare(q){let p=[];const s={q,get p(){return p},bind(...v){p=v;return s},async first(){return native.prepare(q).get(...p)||null},async all(){return{results:native.prepare(q).all(...p)}},async run(){native.prepare(q).run(...p);return{success:true}}};return s},async batch(statements){native.exec('BEGIN');try{for(const s of statements)native.prepare(s.q).run(...s.p);native.exec('COMMIT')}catch(e){native.exec('ROLLBACK');throw e}}};
 return{native,db,ids};
}
const request=body=>new Request('https://ca.example/api/internal/sync',{method:'POST',body:JSON.stringify(body)});
test('changed sources remain withheld until every expected document is staged and activated',async()=>{
 const{native,db,ids}=database(),env={CORPUS:db},options={identityVerifier};
 assert.equal((await handleCloudSync(request({operation:'begin',ids,changed:[ids[0]]}),env,options)).status,200);
 assert.equal(native.prepare('SELECT active FROM documents WHERE id=?').get(ids[0]).active,0);
 const catalog={levels:Object.fromEntries(['foundation','intermediate','final'].map(l=>[l,{subjects:[{id:l+'-1'}]}]))};
 assert.equal((await handleCloudSync(request({operation:'finish',catalog}),env,options)).status,409);
 const document=JSON.parse(native.prepare('SELECT metadata FROM documents WHERE id=?').get(ids[0]).metadata);
 document.applicabilitySources=[{url:NOTICE_URLS[0],recheck:'blocked'}];
 const chunks=[{page:1,page_end:1,text:'An official passage with sufficient source text for an actual staging transaction test.'}];
 assert.equal((await handleCloudSync(request({operation:'put',document,chunks}),env,options)).status,200);
 assert.equal(native.prepare('SELECT active FROM documents WHERE id=?').get(ids[0]).active,0);
 const noticeVerifications=NOTICE_URLS.map(url=>({url,sha256:'b'.repeat(64),verifiedAt:new Date().toISOString(),transport:'cloudflare-relay'}));
 assert.equal((await handleCloudSync(request({operation:'finish',catalog,noticeVerifications}),env,options)).status,200);
 assert.equal(native.prepare('SELECT active FROM documents WHERE id=?').get(ids[0]).active,1);
 const status=JSON.parse(native.prepare("SELECT value FROM sync_state WHERE key='status'").get().value);assert.equal(status.updater.provider,'github-actions');assert.equal(status.documents.length,500);
 assert.deepEqual(JSON.parse(native.prepare('SELECT metadata FROM documents WHERE id=?').get(ids[0]).metadata).applicabilitySources,noticeVerifications);
});
test('a valid workload cannot stage an unofficial source',async()=>{
 const{native,db,ids}=database(),env={CORPUS:db},options={identityVerifier};await handleCloudSync(request({operation:'begin',ids,changed:[ids[0]]}),env,options);
 const document=JSON.parse(native.prepare('SELECT metadata FROM documents WHERE id=?').get(ids[0]).metadata);document.url='https://attacker.example/a.pdf';
 assert.equal((await handleCloudSync(request({operation:'put',document,chunks:[{page:1,page_end:1,text:'This is untrusted text that must not enter the knowledge corpus.'}]}),env,options)).status,400);
});
