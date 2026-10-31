import {spawn} from 'node:child_process';
import {appendFile,mkdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url));
process.chdir(root);
await mkdir('knowledge/cache',{recursive:true});
async function run(command,args){
 await appendFile('knowledge/cache/scheduled-update.log','\n'+new Date().toISOString()+' '+command+' '+args.join(' ')+'\n');
 return new Promise((resolve,reject)=>{
  const child=spawn(command,args,{cwd:root,windowsHide:true,stdio:['ignore','pipe','pipe']});
  for(const stream of [child.stdout,child.stderr])stream.on('data',data=>{process.stdout.write(data);appendFile('knowledge/cache/scheduled-update.log',data).catch(()=>{})});
  child.on('error',reject);child.on('exit',code=>code===0?resolve():reject(Error(args[0]+' failed; see scheduled-update.log')));
 });
}
const python=process.env.CA_PYTHON||(process.platform==='win32'?'venv/Scripts/python.exe':'python3');
await run(process.execPath,['node_modules/wrangler/bin/wrangler.js','whoami']);
await run(process.execPath,['scripts/discover.mjs']);
await run(process.execPath,['scripts/verify-sources.mjs']);
await run(python,['scripts/verify_supplements.py']);
await run(python,['-u','scripts/build_corpus.py']);
await run(process.execPath,['scripts/upload-corpus.mjs','--bulk']);
await run(process.execPath,['scripts/publish-secrets.mjs']);
await run(process.execPath,['node_modules/wrangler/bin/wrangler.js','deploy']);
console.log('ICAI daily source refresh and deployment completed.');
