import {test} from 'node:test';
import assert from 'node:assert/strict';
import {officialURL,validAttempt,eligible,retrieve,systemPrompt} from '../lib/knowledge.js';
const selection={level:'intermediate',attempt:'May 2026',subject:'intermediate-3'};
const meta={schema:'ca-v4',corpus:'active',approved:true,...selection,subjectId:selection.subject,url:'https://resource.cdn.icai.org/material.pdf',text:'This is a source excerpt with sufficient content for testing.',title:'Source',type:'study_material'};
test('only official HTTPS sources are accepted',()=>{assert.equal(officialURL(meta.url),true);for(const u of ['https://icai.org.evil.com/a','http://www.icai.org/a','javascript:alert(1)','https://evil-icai.org/a'])assert.equal(officialURL(u),false)});
test('explicit attempts required',()=>{assert.equal(validAttempt('May 2027'),true);for(const a of ['Latest','All Upcoming','May 27',''])assert.equal(validAttempt(a),false)});
test('future, unapproved, unknown-level and legacy vectors are rejected',()=>{assert.equal(eligible(meta,selection),true);for(const change of [{attempt:'May 2027'},{approved:false},{level:'all'},{schema:'legacy'},{url:'https://example.com/a'},{subjectId:'intermediate-4'}])assert.equal(eligible({...meta,...change},selection),false)});
test('no corpus means no retrieval or speculative fallback',async()=>{assert.deepEqual(await retrieve('GST',selection,{}),[])});
test('filters are applied before semantic retrieval and checked afterward',async()=>{
 let filter;
 const env={CA_KNOWLEDGE:{get:async()=>({corpus:'active'})},AI:{run:async()=>({data:[[0.1]]})},VECTORIZE:{query:async(v,o)=>{filter=o.filter;return{matches:[{score:.9,metadata:meta},{score:.95,metadata:{...meta,attempt:'May 2027'}},{score:.99,metadata:{...meta,schema:'legacy'}}]}}}};
 const r=await retrieve('tax',selection,env);
 assert.deepEqual(filter,{schema:'ca-v4',corpus:'active',level:selection.level,attempt:selection.attempt,subjectId:selection.subject});assert.equal(r.length,1);assert.equal(r[0].attempt,'May 2026');
});
test('prompt requires sources and attempt applicability',()=>{const p=systemPrompt(selection,[meta],'explain','hinglish');assert.match(p,/May 2026/);assert.match(p,/Hinglish/);assert.match(p,/never simply the newest law/);assert.match(p,/untrusted/)});

