import {appliesTo,attemptKey} from './applicability.js';
const stop=new Set('a an the and or for to of in on is are was were be been being can could should would will please explain detail detailed concept question me my i you your with from this that these those how what why when where give show tell solve about according ca icai bhaiya exam attempt selected preparing example examples worked depth does do did into also it its under need help understand ka ki ke hai hain kya ko mein mujhe samjhao aur se'.split(' '));
export function searchTerms(query){return [...new Set((query.toLowerCase().match(/[a-z][a-z0-9]{1,}|\d+[a-z]*/g)||[]).filter(w=>!stop.has(w)))].slice(0,14)}
export function excerpt(text,terms,max=1800){
 const lower=text.toLowerCase();let best=0,score=-1;
 for(let i=0;i<text.length;i+=800){const part=lower.slice(i,i+max);const n=terms.reduce((s,t)=>s+(part.includes(t)?1:0),0);if(n>score){score=n;best=i}}
 return text.slice(Math.max(0,best-160),best+max);
}
export async function retrieveSQL(query,selection,db){
 const terms=searchTerms(query);if(!terms.length)return[];
 const predicate=`d.active=1 AND json_extract(d.metadata,'$.approved')=1 AND d.level=? AND (?='' OR d.subject_id=? OR (?='final-6' AND d.subject_id LIKE 'final-%')) AND (
 (json_extract(d.metadata,'$.applicability.kind')='exact' AND EXISTS(SELECT 1 FROM json_each(json_extract(d.metadata,'$.applicability.attempts')) a WHERE a.value=?)) OR
 (json_extract(d.metadata,'$.applicability.kind')='from' AND json_extract(d.metadata,'$.applicability.from')<=? AND (json_extract(d.metadata,'$.applicability.until') IS NULL OR json_extract(d.metadata,'$.applicability.until')>?)))`;
 const sql=`SELECT c.page,c.page_end,c.text,d.metadata,bm25(chunks_fts,4,1) AS rank FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid JOIN documents d ON d.id=c.document_id WHERE chunks_fts MATCH ? AND ${predicate} ORDER BY rank LIMIT 24`;
 const params=[selection.level,selection.subject||'',selection.subject||'',selection.subject||'',selection.attempt,attemptKey(selection.attempt),attemptKey(selection.attempt)];
 const quote=t=>'"'+t.replaceAll('"','')+'"';
 let rows=(await db.prepare(sql).bind(terms.slice(0,7).map(quote).join(' AND '),...params).all()).results||[];
 if(rows.length<4){const broad=(await db.prepare(sql).bind(terms.map(quote).join(' OR '),...params).all()).results||[];const seen=new Set(rows.map(r=>r.metadata+'|'+r.page));rows.push(...broad.filter(r=>!seen.has(r.metadata+'|'+r.page)))}
 const seen=new Set(),sources=[];
 for(const row of rows){
  let meta;try{meta=JSON.parse(row.metadata)}catch{continue}
  let official=false;try{const u=new URL(meta.url);official=u.protocol==='https:'&&(u.hostname==='icai.org'||u.hostname.endsWith('.icai.org'))}catch{}
  if(!official||meta.approved!==true||meta.level!==selection.level||!appliesTo(meta.applicability,selection.attempt))continue;
  if(selection.subject&&meta.subjectId!==selection.subject&&!(selection.subject==='final-6'&&meta.subjectId.startsWith('final-')))continue;
  const key=meta.url+'|'+row.page;if(seen.has(key))continue;seen.add(key);
  const text=excerpt(row.text,terms);
  const relevance=terms.filter(t=>new RegExp('\\b'+t+'\\b','i').test(text)).length;
  if(!relevance)continue;
  const standard=/\b(ind\s*as|as|sa)\s*[-:]?\s*(\d+)\b/i.exec(query);
  const target=standard?new RegExp(standard[1].toLowerCase()==='as'?'(?:\\bAS|Accounting Standard)\\s*'+standard[2]+'\\b':'\\b'+standard[1].replace(/\s+/g,'\\s*')+'\\s*'+standard[2]+'\\b','i'):null;
  const titleScore=terms.filter(t=>new RegExp('\\b'+t+'\\b','i').test(meta.title)).length/terms.length;
  const specificity=target?(target.test(meta.title)?0.8:target.test(text)?0.2:0):0;
  sources.push({...meta,page:row.page,pageEnd:row.page_end,text,score:relevance/terms.length+titleScore*.25+specificity,attempt:selection.attempt});
 }
 // Relevance first, with authoritative corrections used to resolve matched conflicts.
 const priority=s=>s.score+(['amendment','corrigendum'].includes(s.type)?0.08:0);
 sources.sort((a,b)=>priority(b)-priority(a));
 // When the student explicitly names one standard, avoid filling the context with
 // unrelated chapters that merely contain the same numbers or inventory words.
 const references=[...query.matchAll(/\b(ind\s*as|accounting\s*standard|as|sa)\s*[-:]?\s*(\d+)\b/gi)];
 if(references.length===1&&!/\b(compare|comparison|difference|versus|vs)\b/i.test(query)){
  const [_,family,number]=references[0];
  const pattern=/^(as|accounting\s*standard)$/i.test(family)?'(?:\\bAS|Accounting Standard)\\s*'+number+'\\b':'\\b'+family.replace(/\s+/g,'\\s*')+'\\s*'+number+'\\b';
  const target=new RegExp(pattern,'i');
  const matching=sources.filter(s=>target.test(s.title));
  if(matching.length){const urls=new Set(matching.map(s=>s.url));return sources.filter(s=>urls.has(s.url)||(['amendment','corrigendum'].includes(s.type)&&target.test(s.text))).slice(0,6)}
 }
 return sources.slice(0,6);
}
