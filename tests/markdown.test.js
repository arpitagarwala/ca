import {test} from 'node:test';
import assert from 'node:assert/strict';
import {renderMarkdown} from '../public/markdown.js';
test('model HTML, handlers and links cannot execute',()=>{const r=renderMarkdown('<img src=x onerror=alert(1)>\n[bad](javascript:alert(1))');assert.ok(!r.includes('<img'));assert.ok(!r.includes('<a '));assert.match(r,/&lt;img/)});
test('supported answer formatting renders headings and tables',()=>{const r=renderMarkdown('## Concept\n**Important**\n| A | B |\n|---|---|\n| 1 | 2 |');assert.match(r,/<h3>Concept/);assert.match(r,/<strong>Important/);assert.match(r,/<table>/);assert.match(r,/<td>2/)});
test('journal entries retain alignment and escape HTML inside code fences',()=>{const r=renderMarkdown('```text\nDr Expense    500\n  Cr Inventory  500\n<img onerror=alert(1)>\n```');assert.match(r,/<pre><code>Dr Expense    500\n/);assert.match(r,/&lt;img/);assert.ok(!r.includes('<img'));assert.ok(!r.includes('```'))});
