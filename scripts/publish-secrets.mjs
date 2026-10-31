import 'dotenv/config';
import {spawn} from 'node:child_process';
const values={};
for(const key of ['GROQ_API_KEY','OPENROUTER_API_KEY']){const value=process.env[key]||process.env[key.replace('_API_KEY','')];if(value?.trim())values[key]=value.trim()}
if(!Object.keys(values).length){console.log('No optional provider keys configured. Workers AI remains available.');process.exit(0)}
await new Promise((resolve,reject)=>{
 const child=spawn(process.execPath,['node_modules/wrangler/bin/wrangler.js','secret','bulk'],{stdio:['pipe','inherit','inherit'],windowsHide:true});
 child.stdin.end(JSON.stringify(values));
 child.on('error',reject);child.on('exit',code=>code===0?resolve():reject(Error('Could not publish provider secrets')));
});
