import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const m=JSON.parse(await readFile('knowledge/manifest.json','utf8'));
const c=JSON.parse(await readFile('public/catalog.json','utf8'));
for(const course of Object.values(c.levels))for(const s of course.subjects)for(const resource of s.resources){
 if(!/initial pages/i.test(resource.title)||m.documents.some(d=>d.subjectId===s.id&&d.url===resource.url))continue;
 const base=m.documents.find(d=>d.subjectId===s.id&&d.editionURL===resource.editionURL&&d.type==='study_material');if(!base)continue;
 const r=base.applicability,key=s.id+'|'+resource.url+'|'+r.kind+'|'+r.from+'|'+r.until+'|'+r.attempts.join(',');
 m.documents.push({...base,id:createHash('sha256').update(key).digest('hex').slice(0,32),title:s.title+' — edition, applicability and initial pages',url:resource.url,type:'applicability'});
}
await writeFile('knowledge/manifest.json',JSON.stringify(m,null,2));
console.log('Including edition/applicability pages: '+m.documents.length+' documents');
