// Fail over between independent services within each service's published allowance.
// No API-key rotation, paid-model fallback, or rate-limit bypass.
export class CapacityError extends Error {constructor(message='The free AI allowance is temporarily exhausted. Please try again later.'){super(message);this.name='CapacityError'}}
export class SourceAnswerError extends Error {constructor(){super('The AI could not produce a properly sourced answer. Please rephrase your question.');this.name='SourceAnswerError'}}
export async function reserve(db,key,amount,limit){
 if(!db)throw new CapacityError('Persistent usage controls are unavailable.');
 const row=await db.prepare('INSERT INTO budgets(key,used) SELECT ?,? WHERE ?<=? ON CONFLICT(key) DO UPDATE SET used=budgets.used+excluded.used WHERE budgets.used+excluded.used<=? RETURNING used').bind(key,amount,amount,limit,limit).first();
 return Boolean(row);
}
function retryAt(value,now){const seconds=Number(value);if(value&&Number.isFinite(seconds))return now+Math.max(1,seconds)*1000;const date=Date.parse(value||'');return Number.isFinite(date)?date:now+60000}
export async function generateAnswer(messages,env,options={}){
 const db=env.CORPUS,now=Date.now(),day=new Date(now).toISOString().slice(0,10),minute=Math.floor(now/60000);
 const fetcher=options.fetcher||fetch;
 const text=messages.map(m=>m.content).join('\n');
 const bytes=new TextEncoder().encode(text).length;
 const maxOutput=2300;
 const providers=[];
 let invalidAnswer=false;
 if(env.GROQ_API_KEY||env.GROQ){
  const base={id:'groq',url:'https://api.groq.com/openai/v1/chat/completions',key:env.GROQ_API_KEY||env.GROQ,daily:900,tokens:Math.ceil(bytes/3)+maxOutput};
  providers.push({...base,model:'openai/gpt-oss-120b',effort:'low'});
  providers.push({...base,model:'qwen/qwen3.8-27b',effort:'none'});
 }
 // A paid Cloudflare account can bill above its shared free allocation. An app counter
 // cannot prevent account-wide overages, so free-only deployments must keep it disabled
 // unless the operator has independently confirmed a Workers FREE plan (hard stop).
 if(env.AI&&env.CF_AI_ENABLED==='true'&&(env.FREE_ONLY==='false'||env.CF_FREE_PLAN_CONFIRMED==='true')){
  providers.push({id:'cloudflare-nemotron',model:'@cf/nvidia/nemotron-3-120b-a12b',cf:true,cost:Math.ceil(bytes*45455/1e6+maxOutput*136364/1e6)+30});
  providers.push({id:'cloudflare-gemma',model:'@cf/google/gemma-4-26b-a4b-it',cf:true,cost:Math.ceil(bytes*9091/1e6+maxOutput*27273/1e6)+30});
 }
 if(env.OPENROUTER_API_KEY||env.OPENROUTER)providers.push({id:'openrouter',url:'https://openrouter.ai/api/v1/chat/completions',key:env.OPENROUTER_API_KEY||env.OPENROUTER,model:'openrouter/free',daily:45});
 // Future paid upgrades require the operator to explicitly turn off free-only mode.
 if(env.FREE_ONLY==='false'&&env.CHAT_API_KEY&&env.CHAT_MODEL)providers.push({id:'custom',url:env.CHAT_API_URL,key:env.CHAT_API_KEY,model:env.CHAT_MODEL,daily:1000});
 for(const p of providers){
  const healthKey=p.id+':'+p.model;
  const health=await db.prepare('SELECT retry_at FROM provider_health WHERE provider=?').bind(healthKey).first();
  if(health?.retry_at>now)continue;
  if(p.cf){
   if(!await reserve(db,'cf-neurons:'+day,p.cost,Math.min(8000,Number(env.CF_NEURON_BUDGET)||8000)))continue;
  }else{
   if(!await reserve(db,p.id+':day:'+day,1,p.daily))continue;
   if(!await reserve(db,p.id+':minute:'+minute,1,p.id==='groq'?25:15))continue;
   if(p.tokens){
    if(!await reserve(db,'groq:'+p.model+':tokens:'+minute,p.tokens,7500))continue;
    if(!await reserve(db,'groq:daily-tokens:'+day,p.tokens,180000))continue;
   }
  }
  try{
   let result;
   if(p.cf){result=await env.AI.run(p.model,{messages,max_tokens:maxOutput,temperature:0.15,...(p.id==='cloudflare-nemotron'?{chat_template_kwargs:{enable_thinking:false}}:{})})}
   else{
    if(!p.url?.startsWith('https://'))continue;
    const response=await fetcher(p.url,{method:'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer '+p.key},body:JSON.stringify({model:p.model,messages,max_tokens:maxOutput,temperature:0.15,...(p.id==='groq'?{reasoning_effort:p.effort,include_reasoning:false}:{}),...(p.id==='openrouter'?{provider:{max_price:{prompt:0,completion:0}},route:'fallback'}:{})}),signal:AbortSignal.timeout(30000)});
    if(!response.ok){
     const until=response.status===429?retryAt(response.headers.get('Retry-After'),now):response.status===401||response.status===403?now+3600000:now+30000;
     await db.prepare('INSERT INTO provider_health(provider,retry_at) VALUES(?,?) ON CONFLICT(provider) DO UPDATE SET retry_at=excluded.retry_at').bind(healthKey,until).run();
     continue;
    }
    result=await response.json();
   }
   const answer=result?.choices?.[0]?.message?.content||result?.response;
   if(typeof answer!=='string'||!answer.trim())continue;
   // Never expose hidden reasoning as an answer when a provider returns empty content.
   const clean=answer.replace(/<think>[\s\S]*?<\/think>/gi,'').replace(/\[(?:sources?\s*)?(\d+(?:\s*,\s*\d+)*)\]/gi,(_,numbers)=>numbers.split(/\s*,\s*/).map(n=>'['+n+']').join(' ')).trim();
   if(!clean)continue;
   const citations=[...clean.matchAll(/\[(\d+)\]/g)].map(m=>Number(m[1]));
   if(options.sourceCount&&(!citations.length||citations.some(n=>n<1||n>options.sourceCount))){invalidAnswer=true;options.onInvalidAnswer?.({provider:p.id,finish:result?.choices?.[0]?.finish_reason,answer:clean});continue}
   // Reject the observed failure of noticing contradictory figures and calculating anyway.
   if(/but the given (?:opening )?value|we will follow the numbers|implying.{0,100}(?:we will|we'll|follow|use the)/is.test(clean)){invalidAnswer=true;options.onInvalidAnswer?.({provider:p.id,reason:'conflicting_numerical_data',answer:clean});continue}
   return{answer:clean,provider:p.id,model:p.model};
  }catch{
   await db.prepare('INSERT INTO provider_health(provider,retry_at) VALUES(?,?) ON CONFLICT(provider) DO UPDATE SET retry_at=excluded.retry_at').bind(healthKey,now+60000).run();
  }
 }
 if(invalidAnswer)throw new SourceAnswerError();
 throw new CapacityError();
}
