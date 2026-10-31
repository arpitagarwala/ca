import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {DatabaseSync} from 'node:sqlite';
import {retrieveSQL} from '../lib/search.js';
function setup(){
 const native=new DatabaseSync(':memory:');native.exec(readFileSync(new URL('../knowledge/schema.sql',import.meta.url),'utf8'));
 const docs=[['old','Jan 2027',true,'https://resource.cdn.icai.org/old.pdf'],['new','May 2027',true,'https://resource.cdn.icai.org/new.pdf'],['unapproved','Jan 2027',false,'https://resource.cdn.icai.org/bad.pdf'],['unofficial','Jan 2027',true,'https://example.com/a.pdf']];
 for(const [id,attempt,approved,url]of docs){const metadata={level:'intermediate',subjectId:'intermediate-3',title:'Section 87A rebate',url,type:'study_material',approved,applicability:{kind:'exact',attempts:[attempt]}};native.prepare('INSERT INTO documents(id,level,subject_id,title,url,type,sha256,metadata,updated_at) VALUES(?,?,?,?,?,?,?,?,?)').run(id,'intermediate','intermediate-3',metadata.title,url,'study_material','hash',JSON.stringify(metadata),'today');native.prepare('INSERT INTO chunks(document_id,page,page_end,title,text) VALUES(?,?,?,?,?)').run(id,1,2,metadata.title,'Section 87A rebate applies subject to the conditions specified in this official source excerpt for the selected examination.');}
 return{prepare(sql){return{bind(...params){return{async all(){return{results:native.prepare(sql).all(...params)}}}}}}};
}
test('FTS retrieval isolates attempt, level and approval with actual SQL',async()=>{
 const db=setup(),s={level:'intermediate',subject:'intermediate-3',attempt:'Jan 2027'};
 const old=await retrieveSQL('Section 87A rebate',s,db);assert.equal(old.length,1);assert.equal(old[0].url,'https://resource.cdn.icai.org/old.pdf');
 const next=await retrieveSQL('Section 87A rebate',{...s,attempt:'May 2027'},db);assert.equal(next.length,1);assert.equal(next[0].url,'https://resource.cdn.icai.org/new.pdf');
 assert.deepEqual(await retrieveSQL('Section 87A rebate',{...s,level:'foundation'},db),[]);
});
