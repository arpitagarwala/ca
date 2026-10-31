import {load} from 'cheerio';
import {writeFile} from 'node:fs/promises';
const catalog={checkedAt:new Date().toISOString(),scheme:'New Scheme of Education and Training',levels:{}};
const clean=s=>s.replace(/\s+/g,' ').trim();
async function page(url){url=url.replace(/\/post\.html\?post_id=(\d+)/,'/post/$1');const r=await fetch(url,{signal:AbortSignal.timeout(30000)});if(!r.ok)throw Error(r.status+' '+url);return load(await r.text());}
for(const level of ['foundation','intermediate','final']){
 const url='https://www.icai.org/post/'+level+'-nset', $=await page(url);
 const papers=$('a').toArray().map(a=>({title:clean($(a).text()),url:$(a).attr('href')})).filter(a=>/^(Paper\s*-?\s*\d|Section\s+[AB]:)/i.test(a.title)&&/\/post(?:\/|\.html)/.test(a.url||''));
 catalog.levels[level]={url,subjects:[]};
 for(const [i,p]of papers.entries()){
  const dom=await page(p.url);
  const editions=dom('a').toArray().map(a=>({title:clean(dom(a).text()),url:dom(a).attr('href')})).filter(a=>/Paper\s*-?\s*\d|Section\s+[AB]/i.test(a.title)&&/\/post(?:\/|\.html)/.test(a.url||'')&&a.url!==p.url);
  const pages=[{dom,url:p.url},...await Promise.all(editions.slice(0,6).map(async e=>({dom:await page(e.url),url:e.url})))];
  const links=pages.flatMap(({dom,url})=>dom('a').toArray().map(a=>{try{return{title:clean(dom(a).text()),url:new URL(dom(a).attr('href'),url).href,editionURL:url}}catch{return null}}).filter(a=>a&&/\.pdf(?:\?|$)/i.test(a.url)&&new URL(a.url).hostname.endsWith('.icai.org')));
  const topics=[...new Map(links.filter(a=>/chapter|unit\s*\d/i.test(a.title)).map(a=>[a.title,a])).values()];
  catalog.levels[level].subjects.push({id:level+'-'+(i+1),...p,topics,resources:[...new Map(links.map(a=>[a.url,a])).values()]});
  console.log(level+': '+p.title+' — '+topics.length+' chapter/unit links');
 }
}
await writeFile('public/catalog.json',JSON.stringify(catalog,null,2));
