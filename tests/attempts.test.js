import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parseApplicability,appliesTo,boundEditions} from '../lib/applicability.js';
test('shared years and January rollover follow the printed applicability statement',()=>{
 assert.deepEqual(parseApplicability('(Applicable for May/September 2025 & January 2026 Exams)').attempts,['May 2025','Sep 2025','Jan 2026']);
 assert.deepEqual(parseApplicability('(Applicable for May 2027/ September 2027/ January 2028 Exams)').attempts,['May 2027','Sep 2027','Jan 2028']);
});
test('a newer edition ends the older onwards interval',()=>{
 const editions=boundEditions([{applicability:parseApplicability('(Applicable for May 2026 Exam Onwards)')},{applicability:parseApplicability('(Relevant for May, 2027 and Onwards Examination)')}]);
 assert.equal(appliesTo(editions[0].applicability,'Jan 2027'),true);
 assert.equal(appliesTo(editions[0].applicability,'May 2027'),false);
 assert.equal(appliesTo(editions[1].applicability,'Nov 2026'),false);
});
test('unknown applicability and unlabelled dates cannot be activated',()=>{assert.equal(parseApplicability('Published September 2025'),null);assert.equal(appliesTo(null,'May 2026'),false)});
