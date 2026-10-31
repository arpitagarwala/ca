import {renderMarkdown} from './markdown.js';
const $=s=>document.querySelector(s);
const escapeHTML=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function messageHTML(m){if(m.role==='assistant'&&/but the given (?:opening )?value|we will follow the numbers|implying.{0,100}(?:we will|we'll|follow|use the)/is.test(m.content))return escapeHTML('This earlier answer contained inconsistent numerical data and has been withheld. Please ask again with a complete problem.');return renderMarkdown(m.content)}
let state;try{state=JSON.parse(localStorage.getItem('ca-bhaiya-v4'))}catch{}
if(!state||typeof state!=='object')state={};
state={level:'intermediate',attempt:'Jan 2027',chats:[],saved:[],completed:{},events:[],...state};
if(!['foundation','intermediate','final'].includes(state.level))state.level='intermediate';
for(const k of ['chats','saved','events'])if(!Array.isArray(state[k]))state[k]=[];
if(!state.completed||typeof state.completed!=='object')state.completed={};
let catalog,status,current=null,busy=false,view='tutor';
function persist(){try{localStorage.setItem('ca-bhaiya-v4',JSON.stringify(state))}catch{$('#chatError').textContent='Device storage is full or unavailable. Your current work remains open.'}}
const context=()=>state.level+'|'+state.attempt;
const subjects=()=>catalog?.levels[state.level].subjects||[];
const subjectById=id=>subjects().find(s=>s.id===id);
const api=path=>(document.querySelector('meta[name="ca-api-base"]')?.content||'')+path;
function topicsFor(s){if(!status?.documents)return s.topics;return [...new Map(status.documents.filter(d=>d.subjectId===s.id&&applicable(d)&&(/chapter|unit\s*\d|case study digest/i.test(d.title))).map(d=>[d.title,{title:d.title,url:d.url}])).values()].sort((a,b)=>a.title.localeCompare(b.title,undefined,{numeric:true}));}
function applicable(doc){const r=doc.applicability;if(!r)return doc.attempts?.includes(state.attempt);const [m,y]=state.attempt.split(' ');const key=Number(y)*100+({Jan:1,May:5,Jun:6,Sep:9,Nov:11}[m]||0);return r.kind==='exact'?r.attempts.includes(state.attempt):r.kind==='from'&&key>=r.from&&(!r.until||key<r.until)}
function link(url){try{const u=new URL(url);return u.protocol==='https:'&&(u.hostname==='icai.org'||u.hostname.endsWith('.icai.org'))?escapeHTML(u.href):'#'}catch{return'#'}}
function setView(name){view=name;document.querySelectorAll('.view').forEach(v=>v.hidden=v.id!==name);document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b.dataset.view===name));$('#viewLabel').textContent={tutor:'Ask Bhaiya',curriculum:'Curriculum explorer',amendments:'Amendment tracker',analytics:'My learning',saved:'Saved answers'}[name];if(name==='curriculum')renderCurriculum();if(name==='analytics')renderAnalytics();if(name==='amendments')renderCoverage();if(name==='saved')renderSaved()}
function populateSubjects(){const previous=$('#subject').value;$('#subject').innerHTML='<option value="">All subjects</option>'+subjects().map(s=>'<option value="'+s.id+'">'+escapeHTML(s.title)+'</option>').join('');if(subjectById(previous))$('#subject').value=previous}
function populateAttempts(){const year=new Date().getFullYear();let list=[];for(let y=Math.max(2024,year-2);y<=year+2;y++){
 const months=state.level==='final'?(y===2025?['May','Sep']:y===2026?['Jan','May','Nov']:['May','Nov']):state.level==='foundation'&&y===2024?['Jun','Sep']:y===2024?['May','Sep']:['Jan','May','Sep'];
 for(const m of months)list.push(m+' '+y);
 }if(!list.includes(state.attempt)){state.attempt=state.level==='final'?'Nov '+year:'Jan '+(year+1)}$('#attempt').innerHTML=list.map(a=>'<option>'+escapeHTML(a)+'</option>').join('');$('#attempt').value=state.attempt}
function freshChat(){current=null;$('#question').value='';$('#chatError').textContent='';renderMessages();setView('tutor')}
function renderRecent(){$('#recent').innerHTML=state.chats.filter(c=>c.context===context()).slice(-5).reverse().map(c=>'<button data-chat="'+c.id+'">◌ &nbsp; '+escapeHTML(c.title)+'</button>').join('')}
function renderMessages(){
 const messages=current?.messages||[];$('#welcome').hidden=messages.length>0;$('#starters').hidden=messages.length>0;
 $('#messages').innerHTML=messages.map((m,i)=>'<article class="message '+m.role+'"><div class="message-label">'+(m.role==='user'?'You':'✦ CA Bhaiya')+'</div><div class="answer">'+messageHTML(m)+'</div>'+(m.sources?.length?'<div class="citations">'+m.sources.map(s=>'<a class="citation" href="'+link(s.url)+'" target="_blank" rel="noopener noreferrer">['+s.number+'] '+escapeHTML(s.title)+' · p. '+escapeHTML(s.page||'—')+' ↗</a>').join('')+'</div>':'')+(m.role==='assistant'?'<button class="save-btn" data-save="'+i+'">♧ Save answer</button>':'')+'</article>').join('')+(busy?'<p class="pending" role="status">✧ Finding sources for your selected attempt…</p>':'');
 renderRecent();
}
async function ask(event){
 event?.preventDefault();if(busy||!catalog)return;const query=$('#question').value.trim();if(!query)return;
 if(!current){current={id:crypto.randomUUID(),context:context(),title:query.slice(0,55),messages:[]};state.chats.push(current)}
 const retrying=current.messages.at(-1)?.role==='user'&&current.messages.at(-1)?.content===query;
 const history=(retrying?current.messages.slice(0,-1):current.messages).map(m=>({role:m.role,content:m.content}));
 if(!retrying)current.messages.push({role:'user',content:query});$('#question').value='';busy=true;setBusy(true);$('#chatError').textContent='';persist();renderMessages();
 const selection={level:state.level,attempt:state.attempt,subject:$('#subject').value,mode:$('#mode').value,language:$('#language').value};
 try{
  const r=await fetch(api('/api/chat'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query,history,...selection}),signal:AbortSignal.timeout(90000)});
  const data=await r.json();if(!r.ok)throw Error(data.error||'The request failed. Please retry.');
  current.messages.push({role:'assistant',content:data.answer,sources:data.sources||[],status:data.status});
  state.events.push({context:context(),subject:selection.subject,status:data.status,mode:selection.mode,at:Date.now()});
 }catch(e){$('#chatError').textContent=e.name==='TimeoutError'?'This is taking too long. Your question is saved. Try again.':e.message;$('#question').value=query}
 finally{busy=false;setBusy(false);persist();renderMessages();$('#composer').scrollIntoView({behavior:'smooth',block:'center'})}
}
function setBusy(value){for(const id of ['send','level','attempt','subject','mode','language','newChat'])$('#'+id).disabled=value}
function renderCurriculum(){
 const term=$('#topicSearch').value.toLowerCase();const done=state.completed[context()]||[];
 $('#curriculumList').innerHTML=subjects().map(s=>{
  const topics=topicsFor(s).filter(t=>!term||(s.title+' '+t.title).toLowerCase().includes(term));if(term&&!topics.length&&!s.title.toLowerCase().includes(term))return'';
  return '<details class="subject-card" '+(term?'open':'')+'><summary>'+escapeHTML(s.title)+'<small>'+topics.length+' chapter / unit directory entries · verified for your selected attempt</small></summary><div class="subject-header"><a href="'+link(s.url)+'" target="_blank" rel="noopener noreferrer">Official material & editions ↗</a><span>Self-assessed completion</span></div>'+topics.map(t=>{const key=s.id+'|'+t.title;return'<div class="topic-row"><input type="checkbox" aria-label="Mark topic complete" data-complete="'+escapeHTML(key)+'" '+(done.includes(key)?'checked':'')+'><label>'+escapeHTML(t.title)+'</label><a target="_blank" rel="noopener noreferrer" href="'+link(t.url)+'">ICAI PDF ↗</a><button data-topic="'+escapeHTML(t.title)+'" data-subject="'+s.id+'">Ask →</button></div>'}).join('')+(!topics.length?'<p class="empty">Open the official material page for case studies and resources.</p>':'')+'</details>';
 }).join('')||'<p class="empty">No matching topics. Try another search.</p>';
}
function renderAnalytics(){
 const events=state.events.filter(e=>e.context===context()),done=state.completed[context()]||[],total=subjects().reduce((n,s)=>n+topicsFor(s).length,0);
 $('#analyticsContent').innerHTML='<div class="stats"><div class="stat"><span>Questions asked</span><strong>'+events.length+'</strong><span>For this attempt</span></div><div class="stat"><span>Topics marked complete</span><strong>'+done.length+'</strong><span>Of '+total+' directory entries</span></div><div class="stat"><span>Sourced responses</span><strong>'+events.filter(e=>e.status==='sourced').length+'</strong><span>With official references</span></div></div><div class="panel"><h2>Subject by subject</h2>'+subjects().map(s=>{const completed=done.filter(k=>k.startsWith(s.id+'|')).length;const count=events.filter(e=>e.subject===s.id).length;return'<div class="bar-row"><span>'+escapeHTML(s.title)+'</span><span>'+completed+'/'+topicsFor(s).length+' topics · '+count+' questions</span><div class="bar"><div style="width:'+Math.min(100,100*completed/(topicsFor(s).length||1))+'%"></div></div></div>'}).join('')+'<p>Completion is self-reported. No marks, mastery scores or topic weightages are inferred from chat activity.</p></div>';
}
function renderCoverage(){
 if(!status){$('#coverage').innerHTML='<div class="panel"><h2>Coverage unavailable</h2><p>The source service could not be reached. Try refreshing.</p></div>';return}
 const docs=(status.documents||[]).filter(d=>d.level===state.level&&applicable(d));
 $('#coverage').innerHTML='<div class="panel"><span class="status-tag">'+(docs.length?'Indexed source manifest':'Coverage not confirmed')+'</span><h2 style="margin-top:18px">'+escapeHTML(state.attempt)+' · CA '+escapeHTML(state.level)+'</h2><p>'+docs.length+' verified documents indexed for this attempt. New or changed documents require explicit ICAI applicability evidence before activation.</p><p>Source sync: '+(status.lastChecked?escapeHTML(new Date(status.lastChecked).toLocaleString()):'No completed sync recorded')+'<br>Directory checked: '+escapeHTML(new Date(catalog.checkedAt).toLocaleDateString())+'</p></div>'+docs.map(d=>'<div class="panel"><h2>'+escapeHTML(d.title)+'</h2><p>'+escapeHTML(d.type)+' · '+escapeHTML(d.applicability?.label||d.applicabilityNote||d.applicability?.attempts?.join(', ')||'Verified attempt mapping')+'</p><a href="'+link(d.url)+'" target="_blank" rel="noopener noreferrer">Read official source ↗</a></div>').join('')+'<div class="panel"><h2>Check the official source</h2><p>The latest law is not automatically the law applicable to your examination. Confirm the relevant edition, applicability announcement, statutory updates, RTP and corrigenda together.</p><a href="'+link(catalog.levels[state.level].url)+'" target="_blank" rel="noopener noreferrer">ICAI '+escapeHTML(state.level)+' material ↗</a></div>';
}
function renderSaved(){$('#savedContent').innerHTML=state.saved.filter(s=>s.context===context()).map(s=>'<article class="panel"><div class="message-label">'+escapeHTML(s.title)+'</div><div class="answer">'+messageHTML(s)+'</div><div class="citations">'+(s.sources||[]).map(c=>'<a class="citation" href="'+link(c.url)+'" target="_blank" rel="noopener noreferrer">['+c.number+'] '+escapeHTML(c.title)+' ↗</a>').join('')+'</div><button class="outline-btn" data-unsave="'+s.id+'">Remove</button></article>').join('')||'<p class="empty">Save a helpful answer from your conversation to find it here.<br>Your saved answers are specific to this level and attempt.</p>'}
document.addEventListener('click',e=>{
 const b=e.target.closest('button');if(!b)return;
 if(b.dataset.view)setView(b.dataset.view);
 if(b.id==='newChat')freshChat();
 if(b.dataset.prompt){const selected=subjectById($('#subject').value)||subjects()[0];if(!selected)return;$('#subject').value=selected.id;const topic=topicsFor(selected)[0]?.title||selected.title;const mode=b.textContent.includes('Revise')?'revise':b.textContent.includes('practice')?'practice':'explain';$('#mode').value=mode;$('#question').value=mode==='revise'?'Create a concise revision sheet for '+topic+', with common mistakes and recall questions.':mode==='practice'?'Give one practice question about '+topic+'. Wait for my answer before giving the solution.':'Explain '+topic+' for my attempt with a worked example and common mistakes.';$('#question').focus()}
 if(b.dataset.chat&&!busy){current=state.chats.find(c=>c.id===b.dataset.chat&&c.context===context());renderMessages();setView('tutor')}
 if(b.dataset.topic){if(busy)return;$('#subject').value=b.dataset.subject;$('#question').value='Explain '+b.dataset.topic+' for my selected attempt, with a worked example and common exam mistakes.';setView('tutor');$('#question').focus()}
 if(b.dataset.save!==undefined&&current){const m=current.messages[Number(b.dataset.save)];if(m&&!state.saved.some(s=>s.chatId===current.id&&s.index===Number(b.dataset.save))){state.saved.push({...m,id:crypto.randomUUID(),chatId:current.id,index:Number(b.dataset.save),context:context(),title:current.title});persist()}b.textContent='✓ Saved'}
 if(b.dataset.unsave){state.saved=state.saved.filter(s=>s.id!==b.dataset.unsave);persist();renderSaved()}
});
document.addEventListener('change',e=>{
 if(e.target.matches('[data-complete]')){const k=e.target.dataset.complete;let done=state.completed[context()]||[];state.completed[context()]=e.target.checked?[...new Set([...done,k])]:done.filter(x=>x!==k);persist()}
});
for(const id of ['level','attempt'])$('#'+id).addEventListener('change',()=>{
 state.level=$('#level').value;state.attempt=$('#attempt').value;populateAttempts();persist();populateSubjects();current=null;renderMessages();setView(view);
});
$('#composer').addEventListener('submit',ask);
$('#question').addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();ask()}});
$('#topicSearch').addEventListener('input',renderCurriculum);
$('#level').value=state.level;populateAttempts();setBusy(true);
try{
 const r=await fetch(api('/api/catalog'));if(!r.ok)throw Error('Cannot load the curriculum directory. Refresh to retry.');catalog=await r.json();populateSubjects();renderRecent();
 try{status=await(await fetch(api('/api/status'))).json()}catch{}
}catch(e){$('#chatError').textContent=e.message}
setBusy(!catalog);

