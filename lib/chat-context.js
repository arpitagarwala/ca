import {systemPrompt} from './knowledge.js';
export const MAX_OUTPUT_TOKENS=2300;
export const MAX_INPUT_TOKENS=4900;
export const estimateTokens=messages=>Math.ceil(new TextEncoder().encode(messages.map(m=>m.content).join('\n')).length/3)+messages.length*8+64;
export class QuestionSizeError extends Error {constructor(){super('This question is too long for the free tutor to answer reliably. Please split it into smaller parts, keeping the complete numerical data for each problem.');this.name='QuestionSizeError'}}
export function prepareChatContext(selection,sources,mode,language,history,query){
 const selected=sources.map(s=>({...s})),previous=[...history];
 const build=()=>[{role:'system',content:systemPrompt(selection,selected,mode,language)},...previous,{role:'user',content:query}];
 let messages=build();
 // Preserve the actual question and source numbering. Reduce old conversation
 // first, then lower-ranked excerpts rather than silently skipping the model.
 while(estimateTokens(messages)>MAX_INPUT_TOKENS&&previous.length){previous.shift();messages=build()}
 while(estimateTokens(messages)>MAX_INPUT_TOKENS&&selected.length>1){selected.pop();messages=build()}
 if(estimateTokens(messages)>MAX_INPUT_TOKENS&&selected[0]){
  const excess=(estimateTokens(messages)-MAX_INPUT_TOKENS)*3;
  selected[0].text=selected[0].text.slice(0,Math.max(400,selected[0].text.length-excess-100));messages=build();
 }
 if(estimateTokens(messages)>MAX_INPUT_TOKENS)throw new QuestionSizeError();
 return{messages,sources:selected};
}
