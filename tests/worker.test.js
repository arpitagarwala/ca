import {test} from 'node:test';
import assert from 'node:assert/strict';
import worker from '../worker.js';
const body={query:'Explain this accounting concept',level:'intermediate',attempt:'May 2026',subject:'intermediate-1'};
const metadata={schema:'ca-v4',corpus:'active',approved:true,level:body.level,attempt:body.attempt,subjectId:body.subject,url:'https://resource.cdn.icai.org/test.pdf',title:'Official material',type:'study_material',page:1,text:'Accounting concept excerpt with sufficient supporting explanation for this test.'};
function request(b=body){return new Request('https://ca.example/api/chat',{method:'POST',body:JSON.stringify(b),headers:{'Content-Type':'application/json'}})}
function environment(answer='An explanation supported by the material. [1]') {return {CA_KNOWLEDGE:{get:async()=>({corpus:'active'})},AI:{run:async model=>model.includes('bge')?{data:[[0.1]]}:{response:answer}},VECTORIZE:{query:async()=>({matches:[{score:.9,metadata}]})}}}
test('invalid levels, attempts, subjects and queries return 400',async()=>{for(const b of [{...body,level:'all'},{...body,attempt:'Latest'},{...body,subject:'final-1'},{...body,query:''},null])assert.equal((await worker.fetch(request(b),{})).status,400)});
test('malformed JSON returns 400',async()=>{const r=new Request('https://ca.example/api/chat',{method:'POST',body:'{'});assert.equal((await worker.fetch(r,{})).status,400)});
test('unsupported question receives no fabricated answer or sources',async()=>{const r=await worker.fetch(request(),{}),d=await r.json();assert.equal(d.status,'insufficient_sources');assert.deepEqual(d.sources,[]);assert.match(d.answer,/cannot confirm/)});
test('sourced response carries actual official citation metadata',async()=>{const r=await worker.fetch(request(),environment()),d=await r.json();assert.equal(r.status,200);assert.equal(d.status,'sourced');assert.equal(d.sources[0].url,metadata.url);assert.equal(d.sources[0].page,1);assert.equal(d.sources[0].text,undefined);assert.equal(d.selection.attempt,body.attempt)});
test('fabricated or missing source references are rejected',async()=>{for(const answer of ['Unsupported statement','Wrong citation [9]'])assert.equal((await worker.fetch(request(),environment(answer))).status,502)});
test('AI outage returns a real service error',async()=>{const env=environment();env.AI.run=async model=>{if(model.includes('bge'))return{data:[[0.1]]};throw Error('offline')};assert.equal((await worker.fetch(request(),env)).status,503)});
test('disallowed origins do not reach the tutor',async()=>{const r=request();r.headers.set('Origin','https://evil.example');assert.equal((await worker.fetch(r,environment())).status,403)});
