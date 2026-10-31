import {LEVELS,validAttempt,retrieve,systemPrompt,ANSWER_POLICY_VERSION} from './lib/knowledge.js';
import catalog from './public/catalog.json' with { type: 'json' };
import {generateAnswer,CapacityError,SourceAnswerError,reserve} from './lib/providers.js';
import {handleCloudSync} from './lib/cloud-sync.js';
async function hash(text){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text)))].map(b=>b.toString(16).padStart(2,'0')).join('')}
async function corpusStatus(env){if(env.CORPUS){const row=await env.CORPUS.prepare("SELECT value FROM sync_state WHERE key='status'").first();return row?.value?JSON.parse(row.value):null}return env.CA_KNOWLEDGE?env.CA_KNOWLEDGE.get('ca-v4:status','json'):null}
function json(data,status=200){return Response.json(data,{status,headers:{'Cache-Control':'no-store'}})}
async function handle(request,env){
 const path=new URL(request.url).pathname;
 if(path==='/api/internal/sync')return handleCloudSync(request,env);
 const storedCatalog=env.CORPUS&&(path==='/api/catalog'||path==='/api/chat'||path==='/')?await env.CORPUS.prepare("SELECT value FROM sync_state WHERE key='catalog'").first():null;
 const currentCatalog=storedCatalog?.value?JSON.parse(storedCatalog.value):catalog;
 if(request.method==='GET'&&path==='/api/catalog')return json(currentCatalog);
 if(request.method==='GET'&&path==='/api/status'){
  const status=await corpusStatus(env);
  return json({aiConfigured:Boolean(env.AI||env.GROQ_API_KEY||env.OPENROUTER_API_KEY),retrievalConfigured:Boolean(env.CORPUS),freeOnly:env.FREE_ONLY!=='false',catalogCheckedAt:catalog.checkedAt,...(status||{lastChecked:null,documents:[],message:'Attempt-specific knowledge has not been indexed yet.'})});
 }
 if(path!=='/api/chat'&&path!=='/')return env.ASSETS?env.ASSETS.fetch(request):json({error:'Not found'},404);
 if(request.method!=='POST')return env.ASSETS?env.ASSETS.fetch(request):json({error:'Use POST /api/chat'},405);
 let b;try{const raw=await request.text();if(raw.length>50000)return json({error:'Request too large'},413);b=JSON.parse(raw)}catch{return json({error:'Invalid JSON'},400)}
 if(!b||typeof b.query!=='string'||!b.query.trim()||b.query.length>12000)return json({error:'Ask a question of 1–12,000 characters.'},400);
 if(!LEVELS.includes(b.level)||!validAttempt(b.attempt))return json({error:'Select a CA level and an explicit exam attempt.'},400);
 if(b.subject&&!currentCatalog.levels[b.level].subjects.some(s=>s.id===b.subject))return json({error:'Subject does not belong to selected level.'},400);
 const selection={level:b.level,attempt:b.attempt,subject:b.subject||''};
 const currentStatus=await corpusStatus(env);
 if(env.CORPUS&&(!currentStatus?.lastChecked||Date.now()-Date.parse(currentStatus.lastChecked)>72*3600000))return json({error:'The ICAI source check is overdue. Please consult the official material while the corpus is refreshed.'},503);
 if(env.CORPUS){
  const identity=await hash(request.headers.get('CF-Connecting-IP')||'local');
  if(!await reserve(env.CORPUS,'student:'+identity+':'+Math.floor(Date.now()/3600000),1,20))return json({error:'You have reached the free hourly question allowance. Please try again later.'},429);
 }
 const cacheKey=await hash(JSON.stringify({query:b.query,selection,mode:b.mode,language:b.language,history:b.history,revision:currentStatus?.revision,policy:ANSWER_POLICY_VERSION}));
 if(env.CORPUS){const cached=await env.CORPUS.prepare('SELECT response FROM answer_cache WHERE key=? AND expires>?').bind(cacheKey,Date.now()).first();if(cached?.response)return json({...JSON.parse(cached.response),cached:true})}
 let sources;try{sources=await retrieve(b.query,selection,env)}catch{return json({error:'Source retrieval is unavailable. Please retry shortly.'},503)}
 if(!sources.length)return json({answer:'I don’t have verified source excerpts for this question for '+b.level+', '+b.attempt+' yet. I cannot confirm the applicable provisions or provide a sourced solution. Open the official ICAI material below, or try another indexed subject.',status:'insufficient_sources',sources:[],selection});
 const history=Array.isArray(b.history)?b.history.slice(-4).filter(m=>m&&['user','assistant'].includes(m.role)&&typeof m.content==='string').map(m=>({role:m.role,content:m.content.slice(0,1000)})):[];
 const mode=['explain','revise','practice','compare'].includes(b.mode)?b.mode:'explain';
 const messages=[{role:'system',content:systemPrompt(selection,sources,mode,b.language)},...history,{role:'user',content:b.query}];
 let answer,provider;
 try{
  if(env.CORPUS){({answer,provider}=await generateAnswer(messages,env,{sourceCount:sources.length}))}
  else if(env.FREE_ONLY!=='true'&&env.AI){answer=(await env.AI.run('@cf/meta/llama-3.3-70b-instruct-fp8-fast',{messages,temperature:0.15,max_tokens:2300})).response}
  if(typeof answer!=='string'||!answer.trim())throw Error('Empty response');
 }catch(error){return json({error:error instanceof CapacityError||error instanceof SourceAnswerError?error.message:'The tutor is temporarily unavailable. Your question is saved; please retry.'},error instanceof CapacityError?429:error instanceof SourceAnswerError?502:503)}
 const citations=[...answer.matchAll(/\[(\d+)\]/g)].map(m=>Number(m[1]));
 if(!citations.length||citations.some(n=>n<1||n>sources.length))return json({error:'The answer could not be validated against its sources. Please rephrase or retry.'},502);
 const result={answer,status:'sourced',policyVersion:ANSWER_POLICY_VERSION,selection,provider,sources:sources.map((s,i)=>({number:i+1,title:s.title,url:s.url,type:s.type,page:s.page,pageEnd:s.pageEnd,subject:s.subject,attempt:selection.attempt,sha256:s.sha256}))};
 if(env.CORPUS)await env.CORPUS.prepare('INSERT INTO answer_cache(key,response,expires) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET response=excluded.response,expires=excluded.expires').bind(cacheKey,JSON.stringify(result),Date.now()+86400000).run();
 return json(result);
}
export default{
 async fetch(request,env){
  const origin=request.headers.get('Origin'),allowed=(env.ALLOWED_ORIGIN||'').split(',').map(s=>s.trim());
  if(origin&&origin!==new URL(request.url).origin&&!allowed.includes(origin))return json({error:'Origin not allowed'},403);
  let response;try{response=request.method==='OPTIONS'?new Response(null,{status:204}):await handle(request,env)}catch(e){console.error('Request failed',e.name);response=json({error:'Service temporarily unavailable'},503)}
  const headers=new Headers(response.headers);
  if(origin){headers.set('Access-Control-Allow-Origin',origin);headers.set('Vary','Origin');headers.set('Access-Control-Allow-Methods','GET, POST, OPTIONS');headers.set('Access-Control-Allow-Headers','Content-Type')}
  headers.set('X-Content-Type-Options','nosniff');headers.set('Referrer-Policy','strict-origin-when-cross-origin');
  return new Response(response.body,{status:response.status,headers});
 }
};
