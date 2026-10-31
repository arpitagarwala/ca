import 'dotenv/config';
import {writeFile} from 'node:fs/promises';
import worker from '../worker.js';
import {sql} from './cloudflare.mjs';
const database='79a2cca9-859f-45a8-ac54-77fb7eaa76a0';
const db={prepare(query){let params=[];const statement={bind(...values){params=values;return statement},async first(){return(await sql(database,query,params))[0].results?.[0]||null},async all(){return{results:(await sql(database,query,params))[0].results||[]}},async run(){return(await sql(database,query,params))[0]}};return statement}};
const env={CORPUS:db,FREE_ONLY:'true',CF_AI_ENABLED:'false',GROQ_API_KEY:process.env.GROQ_API_KEY||process.env.GROQ,OPENROUTER_API_KEY:process.env.OPENROUTER_API_KEY||process.env.OPENROUTER};
const cases=[
 {level:'foundation',attempt:'May 2027',subject:'foundation-2',query:'Explain contract of guarantee and the rights of a surety under the Indian Contract Act. Give a short example.'},
 {level:'intermediate',attempt:'Jan 2027',subject:'intermediate-1',query:'Explain lower of cost and net realisable value under AS 2 with one numerical example.'},
 {level:'final',attempt:'Nov 2026',subject:'final-1',query:'Explain recognition of deferred tax assets for unused tax losses under Ind AS 12.'}
];
const results=[];
for(const c of cases){
 const started=Date.now();const response=await worker.fetch(new Request('https://ca.example/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...c,mode:'explain',language:'english'})}),env);
 const data=await response.json();results.push({...c,httpStatus:response.status,...data,durationMs:Date.now()-started});
 console.log(JSON.stringify({level:c.level,http:response.status,status:data.status,provider:data.provider,sources:data.sources?.length,error:data.error,answerPreview:data.answer?.slice(0,220)}));
}
await writeFile('knowledge/eval-report.json',JSON.stringify({testedAt:new Date().toISOString(),results},null,2));
if(results.some(r=>r.status!=='sourced'))process.exitCode=1;
