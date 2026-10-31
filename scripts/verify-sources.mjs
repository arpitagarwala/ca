import {load} from 'cheerio';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {parseApplicability,boundEditions} from '../lib/applicability.js';
import {officialURL} from '../lib/knowledge.js';
const catalog=JSON.parse(await readFile('public/catalog.json','utf8'));
const clean=s=>(s||'').replace(/\s+/g,' ').trim();
const cache=new Map();
async function page(url){
 url=url.replace(/\/post\.html\?post_id=(\d+)/,'/post/$1');
 if(cache.has(url))return cache.get(url);
 for(let i=0;i<3;i++){try{const r=await fetch(url,{signal:AbortSignal.timeout(30000)});if(!r.ok)throw Error('HTTP '+r.status);const text=await r.text();const d=load(text);cache.set(url,d);return d}catch(e){if(i===2)throw e}}
}
function labelFor(d,anchor){
 const own=clean(d(anchor).text());if(parseApplicability(own))return own;
 let element=d(anchor);
 for(let n=0;n<7&&element.length;n++,element=element.parent()){
  const direct=clean(element.clone().children('ul,ol,table').remove().end().text());
  const candidate=direct.match(/\((?:Applicable|Relevant)[^)]*\)/i)?.[0];
  if(candidate&&parseApplicability(candidate))return candidate;
  const prev=clean(element.prev().text());if(prev.length<350&&parseApplicability(prev))return prev;
 }
 return '';
}
const manifest={version:2,checkedAt:new Date().toISOString(),verification:'official-origin-and-published-applicability; not human academic review',documents:[],pending:[],errors:[]};
for(const [level,course]of Object.entries(catalog.levels))for(const subject of course.subjects){
 try{
 const root=await page(subject.url);
 let editions=root('a').toArray().map(a=>({url:root(a).attr('href'),label:labelFor(root,a),title:clean(root(a).text())})).filter(a=>a.url&&officialURL(a.url)&&/\/post(?:\/|\.html)/.test(a.url)&&/Paper\s*-?\s*\d|Section\s+[AB]/i.test(a.title));
 editions=boundEditions(editions.map(e=>({...e,applicability:parseApplicability(e.label)})));
 // The root can contain amendments with their own exact attempt labels.
 const pages=[{url:subject.url,dom:root,applicability:null,label:''},...await Promise.all(editions.map(async e=>({...e,dom:await page(e.url)})))];
 const seen=new Set();
 for(const p of pages){
  const titleLabel=clean(p.dom('ol.breadcrumb').text());
  const inherited=p.applicability||parseApplicability(titleLabel);
  const article=p.dom('.table-responsive').first();
  const anchors=(article.length?article:p.dom('body')).find('a').toArray();
  for(const a of anchors){
   let url;try{url=new URL(p.dom(a).attr('href'),p.url).href}catch{continue}
   if(!officialURL(url)||! /\.pdf(?:\?|$)/i.test(url))continue;
   const title=clean(p.dom(a).text());if(!title||/^(contents|index pages)$/i.test(title))continue;
   const ownLabel=labelFor(p.dom,a),ownRule=parseApplicability(ownLabel);
   const type=/corrigendum|errat/i.test(title)?'corrigendum':/amendment|statutory|judicial|update/i.test(title)?'amendment':/case study|case studies|digest/i.test(title)?'case_study':/rtp|revision test|mock test|model test/i.test(title)?'practice':'study_material';
   const rule=(ownRule&&(!inherited||type==='amendment')?ownRule:inherited||ownRule);
   const key=url+'|'+JSON.stringify(rule);if(seen.has(key))continue;seen.add(key);
   const usesOwn=rule===ownRule;
   const doc={id:createHash('sha256').update(level+'|'+subject.id+'|'+key).digest('hex').slice(0,32),level,subjectId:subject.id,subject:subject.title,title,url,type,applicability:rule,applicabilitySource:usesOwn?p.url:subject.url,editionURL:p.url,applicabilityNote:usesOwn?ownLabel:p.label||titleLabel,verification:'official_source',approved:Boolean(rule)};
   if(rule)manifest.documents.push(doc);else manifest.pending.push({...doc,reason:'No explicit exam-applicability statement found'});
  }
 }
 console.log(level+' '+subject.id+': '+manifest.documents.filter(d=>d.subjectId===subject.id).length+' applicable resources');
 }catch(e){manifest.errors.push({subjectId:subject.id,message:e.message});console.log(subject.id+': source check failed')}
}
// A newer edition ends an older 'onwards' edition even when an old page title remains broad.
for(const doc of manifest.documents)if(doc.applicability.kind==='from'){
 const newer=manifest.documents.filter(d=>d.subjectId===doc.subjectId&&d.applicability.kind==='from'&&d.applicability.from>doc.applicability.from).map(d=>d.applicability.from);
 if(newer.length)doc.applicability.until=Math.min(...newer);
}
manifest.documents=[...new Map(manifest.documents.map(d=>{const r=d.applicability;const key=d.subjectId+'|'+d.url+'|'+r.kind+'|'+r.from+'|'+r.until+'|'+r.attempts.join(',');return[key,{...d,id:createHash('sha256').update(key).digest('hex').slice(0,32)}]})).values()];
await mkdir('knowledge',{recursive:true});
await writeFile('knowledge/manifest.json',JSON.stringify(manifest,null,2));
console.log(JSON.stringify({documents:manifest.documents.length,pending:manifest.pending.length,errors:manifest.errors.length}));
if(manifest.errors.length)process.exitCode=1;
