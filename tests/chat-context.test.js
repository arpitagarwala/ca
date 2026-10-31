import {test} from 'node:test';
import assert from 'node:assert/strict';
import {prepareChatContext,estimateTokens,MAX_INPUT_TOKENS,MAX_OUTPUT_TOKENS,QuestionSizeError} from '../lib/chat-context.js';
import {systemPrompt} from '../lib/knowledge.js';
const selection={level:'final',attempt:'Nov 2026',subject:'final-1'};
const sources=Array.from({length:6},(_,i)=>({title:'Official source '+i,url:'https://resource.cdn.icai.org/source'+i+'.pdf',page:i+1,attempt:'Nov 2026',type:i===0?'amendment':'study_material',text:'Relevant official provision. '.repeat(72)}));
test('large sourced conversations fit the free model without dropping the student problem',()=>{
 const query='Explain recognition, measurement and exceptions using this problem: '+ 'Complete problem data. '.repeat(70);
 const history=Array.from({length:4},(_,i)=>({role:i%2?'assistant':'user',content:'Earlier conversation. '.repeat(50)}));
 const before=[{role:'system',content:systemPrompt(selection,sources,'explain','english')},...history,{role:'user',content:query}];
 assert.ok(estimateTokens(before)+MAX_OUTPUT_TOKENS>7500);
 const result=prepareChatContext(selection,sources,'explain','english',history,query);
 assert.ok(estimateTokens(result.messages)<=MAX_INPUT_TOKENS);
 assert.equal(result.messages.at(-1).content,query);
 assert.deepEqual(result.sources,sources.slice(0,result.sources.length));
 assert.equal(result.sources[0].type,'amendment');
});
test('oversized Unicode problems get a size error instead of a capacity error',()=>{
 assert.throws(()=>prepareChatContext(selection,sources,'explain','english',[],'प्रश्न '.repeat(2000)),QuestionSizeError);
});
