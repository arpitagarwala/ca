import {test} from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {generateAnswer,reserve,CapacityError,SourceAnswerError} from '../lib/providers.js';
function database(){const db=new DatabaseSync(':memory:');db.exec('CREATE TABLE budgets(key TEXT PRIMARY KEY,used INTEGER NOT NULL);CREATE TABLE provider_health(provider TEXT PRIMARY KEY,retry_at INTEGER NOT NULL);');return{prepare(sql){return{bind(...params){return{async first(){return db.prepare(sql).get(...params)||null},async run(){db.prepare(sql).run(...params);return{success:true}}}}}}}}
const messages=[{role:'user',content:'Explain the concept using source [1].'}];
test('atomic budget reservation never exceeds the allowance',async()=>{const db=database();const r=await Promise.all(Array.from({length:20},()=>reserve(db,'day',2,10)));assert.equal(r.filter(Boolean).length,5)});
test('429 fails over to an independent provider and respects Retry-After',async()=>{
 const db=database();let calls=0;
 const env={CORPUS:db,GROQ_API_KEY:'test-only',AI:{run:async()=>({response:'Supported answer [1]'})},FREE_ONLY:'true',CF_AI_ENABLED:'true',CF_FREE_PLAN_CONFIRMED:'true'};
 const fetcher=async()=>{calls++;return new Response('',{status:429,headers:{'Retry-After':'3600'}})};
 assert.equal((await generateAnswer(messages,env,{fetcher,sourceCount:1})).provider,'cloudflare-nemotron');
 await generateAnswer(messages,env,{fetcher,sourceCount:1});assert.equal(calls,2);
});
test('free-only mode never calls a configured paid custom endpoint',async()=>{
 const env={CORPUS:database(),FREE_ONLY:'true',CHAT_API_KEY:'test-only',CHAT_MODEL:'paid',CHAT_API_URL:'https://paid.example'};
 await assert.rejects(generateAnswer(messages,env,{fetcher:async()=>{throw Error('Must not be called')}}),CapacityError);
});
test('OpenRouter requests use its verified free router and zero-price constraint',async()=>{
 let payload;
 const env={CORPUS:database(),OPENROUTER_API_KEY:'test-only',FREE_ONLY:'true'};
 await generateAnswer(messages,env,{sourceCount:1,fetcher:async(u,o)=>{payload=JSON.parse(o.body);return Response.json({choices:[{message:{content:'Answer [1]'}}]})}});
 assert.equal(payload.model,'openrouter/free');assert.deepEqual(payload.provider.max_price,{prompt:0,completion:0});
});
test('unsupported citation numbers cause validation failure',async()=>{
 await assert.rejects(generateAnswer(messages,{CORPUS:database(),AI:{run:async()=>({response:'Unsupported [99]'})},CF_AI_ENABLED:'true',CF_FREE_PLAN_CONFIRMED:'true'},{sourceCount:1}),SourceAnswerError);
});
test('Cloudflare inference is disabled on unconfirmed/paid plans in free-only mode',async()=>{let called=false;await assert.rejects(generateAnswer(messages,{CORPUS:database(),FREE_ONLY:'true',CF_AI_ENABLED:'true',CF_FREE_PLAN_CONFIRMED:'false',AI:{run:async()=>{called=true}}}),CapacityError);assert.equal(called,false)});
test('valid combined citations are normalized and a failed model can use the second free model',async()=>{
 const models=[];const r=await generateAnswer(messages,{CORPUS:database(),GROQ_API_KEY:'test-only',FREE_ONLY:'true'},{sourceCount:2,fetcher:async(url,options)=>{const m=JSON.parse(options.body).model;models.push(m);return Response.json({choices:[{message:{content:models.length===1?'No citations':'Supported [Sources 1, 2]'}}]})}});
 assert.equal(r.model,'qwen/qwen3.8-27b');assert.match(r.answer,/\[1\] \[2\]/);
});
test('observed numerical-conflation failure is rejected before showing an answer',async()=>{
 let calls=0;const r=await generateAnswer(messages,{CORPUS:database(),GROQ_API_KEY:'test-only',FREE_ONLY:'true'},{sourceCount:1,fetcher:async()=>Response.json({choices:[{message:{content:++calls===1?'The figures differ, but the given opening value implies another cost; we will follow the numbers. [1]':'Illustrative example — invented data: 100 units at Rs 50 cost and Rs 45 NRV. Inventory = 100 × 45 = Rs 4500. [1]'}}]})});assert.equal(r.model,'qwen/qwen3.8-27b');assert.equal(calls,2);
});
