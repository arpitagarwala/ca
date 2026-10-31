const months={jan:1,january:1,may:5,jun:6,june:6,sep:9,sept:9,september:9,nov:11,november:11};
const names={1:'Jan',5:'May',6:'Jun',9:'Sep',11:'Nov'};
export function attemptKey(a){const m=/^(Jan|May|Jun|Sep|Nov) (20\d{2})$/.exec(a||'');return m?Number(m[2])*100+months[m[1].toLowerCase()]:0}
export function parseApplicability(label){
 if(typeof label!=='string'||!/(applicable|relevant|statutory|amendment|examination|exams?)/i.test(label))return null;
 const tokens=[...label.matchAll(/\b(january|jan|may|june|jun|september|sept|sep|november|nov)\b\s*,?\s*(20\d{2})?/gi)];
 const attempts=[];
 for(let i=0;i<tokens.length;i++){
  const year=tokens[i][2]||tokens.slice(i+1).find(t=>t[2])?.[2]||tokens.slice(0,i).reverse().find(t=>t[2])?.[2];
  if(year)attempts.push(names[months[tokens[i][1].toLowerCase()]]+' '+year);
 }
 const unique=[...new Set(attempts)].sort((a,b)=>attemptKey(a)-attemptKey(b));
 if(!unique.length)return null;
 if(/onwards?|onward|and subsequent/i.test(label))return{kind:'from',from:attemptKey(unique[0]),until:null,attempts:[],label:label.trim()};
 return{kind:'exact',from:null,until:null,attempts:unique,label:label.trim()};
}
export function appliesTo(rule,attempt){const key=attemptKey(attempt);return Boolean(key&&rule&&(rule.kind==='exact'?rule.attempts.includes(attempt):rule.kind==='from'&&key>=rule.from&&(!rule.until||key<rule.until)))}
export function boundEditions(editions){
 const starts=editions.map(e=>e.applicability?.from).filter(Boolean).sort((a,b)=>a-b);
 return editions.map(e=>({...e,applicability:e.applicability?.kind==='from'?{...e.applicability,until:starts.find(n=>n>e.applicability.from)||null}:e.applicability}));
}
