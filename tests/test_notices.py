import pathlib
import sys
import unittest
import pymupdf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from official_notices import verify_notice, SOURCES

def notice(attempt='May', digest='2025'):
    text = '\n'.join([
        'BOS STUDY MATERIAL APPLICABLE FOR FINAL COURSE, ' + attempt + ' 2026 EXAMINATION',
        'Advanced Financial Management November, 2024 (Reprint August, 2025 and onwards)',
        'Study Material of SPOM Set-A (September, 2025)',
        'Study Material of SPOM Set-B (October, 2023)',
        'IBS Case Study Digest (July 2024 edition)',
        'IBS Case Study Digest (July ' + digest + ' edition) VOL II',
    ])
    with pymupdf.open() as pdf:
        pdf.new_page().insert_textbox((30, 30, 570, 800), text, fontsize=10)
        return pdf.tobytes()

class NoticeVerification(unittest.TestCase):
    def test_live_facts_include_hash_time_and_transport(self):
        result = verify_notice(*SOURCES[0], notice(), 'cloudflare-relay')
        self.assertEqual(result['transport'], 'cloudflare-relay')
        self.assertEqual(len(result['sha256']), 64)
        self.assertIn('verifiedAt', result)

    def test_wrong_attempt_and_changed_digest_are_not_accepted(self):
        for data in [notice(attempt='November'), notice(digest='2027'), b'<html>Blocked</html>']:
            with self.assertRaises((RuntimeError, ValueError)):
                verify_notice(*SOURCES[0], data, 'cloudflare-relay')

if __name__ == '__main__':
    unittest.main()
