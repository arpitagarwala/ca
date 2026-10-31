export const SYNC_AUDIENCE='https://ca-chatbot.arpitagarwala.workers.dev/api/internal/sync';
const issuer='https://token.actions.githubusercontent.com';
let cachedKeys=null,keysExpire=0;
const decode=s=>Uint8Array.from(atob(s.replaceAll('-','+').replaceAll('_','/')),c=>c.charCodeAt(0));
export async function verifyGitHubIdentity(authorization,options={}){
 if(!authorization?.startsWith('Bearer '))throw Error('GitHub identity required');
 const token=authorization.slice(7);if(token.length>16000)throw Error('Invalid token');
 const parts=token.split('.');if(parts.length!==3)throw Error('Invalid token');
 const header=JSON.parse(new TextDecoder().decode(decode(parts[0]))),claims=JSON.parse(new TextDecoder().decode(decode(parts[1])));
 if(header.alg!=='RS256'||typeof header.kid!=='string')throw Error('Invalid signing algorithm');
 let keys;
 if(options.fetcher){const r=await options.fetcher(issuer+'/.well-known/jwks');keys=(await r.json()).keys}
 else{
  if(!cachedKeys||Date.now()>keysExpire){const r=await fetch(issuer+'/.well-known/jwks',{cf:{cacheTtl:3600,cacheEverything:true}});if(!r.ok)throw Error('Identity issuer unavailable');cachedKeys=(await r.json()).keys;keysExpire=Date.now()+3600000}
  keys=cachedKeys;
 }
 const jwk=keys?.find(k=>k.kid===header.kid&&k.kty==='RSA');if(!jwk)throw Error('Unknown signing key');
 const key=await crypto.subtle.importKey('jwk',jwk,{name:'RSASSA-PKCS1-v1_5',hash:'SHA-256'},false,['verify']);
 if(!await crypto.subtle.verify('RSASSA-PKCS1-v1_5',key,decode(parts[2]),new TextEncoder().encode(parts[0]+'.'+parts[1])))throw Error('Invalid signature');
 const now=Date.now()/1000;
 if(!Number.isFinite(claims.exp)||claims.exp<=now||!Number.isFinite(claims.nbf)||claims.nbf>now+60||!Number.isFinite(claims.iat)||claims.iat>now+60||claims.exp-claims.iat>900)throw Error('Expired or invalid identity');
 if(claims.iss!==issuer||claims.aud!==SYNC_AUDIENCE||claims.repository!=='arpitagarwala/ca'||String(claims.repository_id)!=='1191403719'||String(claims.repository_owner_id)!=='210027759'||claims.ref!=='refs/heads/main'||claims.workflow_ref!=='arpitagarwala/ca/.github/workflows/knowledge-sync.yml@refs/heads/main'||claims.repository_visibility!=='public'||claims.runner_environment!=='github-hosted'||!['schedule','workflow_dispatch','push'].includes(claims.event_name))throw Error('Workflow not authorized');
 return claims;
}
