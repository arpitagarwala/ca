import {readFile,writeFile} from 'node:fs/promises';
const html=await readFile('public/index.html','utf8');
await writeFile('index.html',html.replace('<head>','<head><meta name="ca-api-base" content="https://ca-chatbot.arpitagarwala.workers.dev">').replace('href="/style.css"','href="/public/style.css"').replace('src="/app.js"','src="/public/app.js"'));
console.log('Prepared GitHub Pages entry point with the deployed Worker API.');
