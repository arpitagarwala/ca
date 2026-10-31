import 'dotenv/config';
import {writeFile} from 'node:fs/promises';
import {sql} from './cloudflare.mjs';
import {retrieveSQL} from '../lib/search.js';
import {systemPrompt} from '../lib/knowledge.js';
import {generateAnswer} from '../lib/providers.js';
const db={prepare(q){let p=[];const s={bind(...v){p=v;return s},async first(){return(await sql('79a2cca9-859f-45a8-ac54-77fb7eaa76a0',q,p))[0].results?.[0]||null},async all(){return{results:(await sql('79a2cca9-859f-45a8-ac54-77fb7eaa76a0',q,p))[0].results||[]}},async run(){return(await sql('79a2cca9-859f-45a8-ac54-77fb7eaa76a0',q,p))[0]}};return s}};
const selection={level:'intermediate',attempt:'May 2027',subject:'intermediate-1'};
const query='Explain Unit 1: Accounting Standard 2 Valuation of Inventory for my selected attempt, with a worked example and common exam mistakes.';
const sources=await retrieveSQL(query,selection,db),invalid=[];
try{
 const r=await generateAnswer([{role:'system',content:systemPrompt(selection,sources,'explain','english')},{role:'user',content:query}],{CORPUS:db,FREE_ONLY:'true',GROQ_API_KEY:process.env.GROQ_API_KEY||process.env.GROQ,OPENROUTER_API_KEY:process.env.OPENROUTER_API_KEY||process.env.OPENROUTER},{sourceCount:sources.length,onInvalidAnswer:a=>invalid.push(a)});
 console.log(JSON.stringify({provider:r.provider,sources:sources.length,chars:r.answer.length,preview:r.answer.slice(0,180)}));
}finally{await writeFile('knowledge/cache/answer-diagnostics.json',JSON.stringify({sources:sources.map(s=>({title:s.title,url:s.url,page:s.page})),invalid},null,2));console.log(invalid.map(x=>({provider:x.provider,finish:x.finish,chars:x.answer.length,begin:x.answer.slice(0,180),end:x.answer.slice(-180)})))}
