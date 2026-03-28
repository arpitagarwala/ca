const fs = require('fs');
const lines = fs.readFileSync('vectors.ndjson', 'utf8').split('\n').filter(l => l.trim().length > 0);
let interCount = 0;
let fndCount = 0;
let finalCount = 0;
lines.forEach(l => {
  const v = JSON.parse(l);
  if (v.metadata) {
    if (v.metadata.level === 'intermediate') interCount++;
    if (v.metadata.level === 'foundation') fndCount++;
    if (v.metadata.level === 'final') finalCount++;
  }
});
console.log('Intermediate:', interCount, 'Foundation:', fndCount, 'Final:', finalCount);

const finalChunks = lines.filter(l => JSON.parse(l).metadata?.level === 'final').slice(0, 3);
finalChunks.forEach(l => {
    const meta = JSON.parse(l).metadata;
    console.log('Sample Final metadata:', meta.source, meta.subject, meta.attempts);
});
