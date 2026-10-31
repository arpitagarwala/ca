import 'dotenv/config';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
export const account=process.env.CLOUDFLARE_ACCOUNT_ID||'bacdfb7c3105ba2451a6dfd467803073';
let token;
async function credential(){
 if(process.env.CLOUDFLARE_API_TOKEN)return process.env.CLOUDFLARE_API_TOKEN;
 const file=path.join(process.env.APPDATA||'', 'xdg.config','.wrangler','config','default.toml');
 const config=await readFile(file,'utf8');
 const expiry=/expiration_time\s*=\s*"([^"]+)"/.exec(config)?.[1];
 if(expiry&&new Date(expiry)<new Date())throw Error('Wrangler login expired; run wrangler whoami to refresh it.');
 const value=/oauth_token\s*=\s*"([^"]+)"/.exec(config)?.[1];
 if(!value)throw Error('Cloudflare authentication not available');return value;
}
export async function cf(route,options={}){
 token||=await credential();
 const r=await fetch('https://api.cloudflare.com/client/v4'+route,{...options,headers:{Authorization:'Bearer '+token,...options.headers},signal:AbortSignal.timeout(60000)});
 let d;try{d=await r.json()}catch{throw Error('Cloudflare returned non-JSON response: '+r.status)}
 if(!r.ok||d.success===false)throw Error('Cloudflare '+r.status+': '+(d.errors||[]).map(e=>e.message).join('; '));
 return d.result;
}
export async function sql(id,query,params=[]){return cf('/accounts/'+account+'/d1/database/'+id+'/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sql:query,params})})}
