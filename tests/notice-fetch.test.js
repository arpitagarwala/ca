import {test} from 'node:test';
import assert from 'node:assert/strict';
import {fetchOfficialNotice,NOTICE_URLS} from '../lib/notice-fetch.js';
import {handleCloudSync} from '../lib/cloud-sync.js';
test('notice relay returns PDF bytes only for the two approved official URLs',async()=>{
 let calls=0;
 const fetcher=async(url,options)=>{calls++;assert.equal(options.redirect,'manual');return new Response('%PDF-1.7\nnotice')};
 assert.equal((await fetchOfficialNotice('https://attacker.example/a.pdf',fetcher)).status,400);
 assert.equal(calls,0);
 const response=await fetchOfficialNotice(NOTICE_URLS[0],fetcher);
 assert.equal(response.status,200);assert.equal(await response.text(),'%PDF-1.7\nnotice');
});
test('notice relay rejects HTTP errors, HTML challenges, redirects and oversized PDFs',async()=>{
 for(const response of [new Response('Blocked',{status:403}),new Response('<html>Challenge</html>'),new Response('',{status:302,headers:{Location:'https://attacker.example'}}),new Response('%PDF-'+'.'.repeat(2000000))]){
  assert.equal((await fetchOfficialNotice(NOTICE_URLS[1],async()=>response)).status,502);
 }
});
test('unauthenticated requests cannot trigger notice fetches',async()=>{
 let called=false;
 const request=new Request('https://example.com/api/internal/sync',{method:'POST',body:JSON.stringify({operation:'fetch-notice',url:NOTICE_URLS[0]})});
 const response=await handleCloudSync(request,{CORPUS:{}},{identityVerifier:async()=>{throw Error('Invalid identity')},noticeFetcher:async()=>{called=true}});
 assert.equal(response.status,401);assert.equal(called,false);
});
