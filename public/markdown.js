// Render the formatting used by the tutor. Escape all source HTML before adding
// a small fixed set of elements; model-provided links/images/HTML never execute.
export function renderMarkdown(value){
 const escape=s=>s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const inline=s=>escape(s).replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*([^*\n]+)\*/g,'<em>$1</em>');
 const lines=String(value||'').split('\n'),out=[];let table=[];
 const flush=()=>{if(!table.length)return;const rows=table.filter(l=>!/^\s*\|?[\s:|\-]+\|?\s*$/.test(l));out.push('<div class="table-scroll"><table>'+rows.map((r,i)=>'<tr>'+r.trim().replace(/^\|/,'').replace(/\|$/,'').split('|').map(c=>'<'+(i?'td':'th')+'>'+inline(c.trim())+'</'+(i?'td':'th')+'>').join('')+'</tr>').join('')+'</table></div>');table=[]};
 for(const line of lines){
  if(/^\s*\|.+\|\s*$/.test(line)){table.push(line);continue}flush();
  if(!line.trim()){out.push('<div class="answer-gap"></div>');continue}
  const h=/^#{1,4}\s+(.+)/.exec(line);if(h){out.push('<h3>'+inline(h[1])+'</h3>');continue}
  if(/^\s*[-*_]{3,}\s*$/.test(line)){out.push('<hr>');continue}
  const bullet=/^\s*[-*]\s+(.+)/.exec(line);if(bullet){out.push('<div class="answer-bullet">• '+inline(bullet[1])+'</div>');continue}
  out.push('<div>'+inline(line)+'</div>');
 }
 flush();return out.join('');
}
