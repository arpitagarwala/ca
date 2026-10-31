PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS documents (
 id TEXT PRIMARY KEY, level TEXT NOT NULL, subject_id TEXT NOT NULL, title TEXT NOT NULL,
 url TEXT NOT NULL, type TEXT NOT NULL, sha256 TEXT NOT NULL, metadata TEXT NOT NULL,
 updated_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS chunks (
 id INTEGER PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 page INTEGER NOT NULL, page_end INTEGER NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS documents_subject ON documents(level,subject_id,active);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(title,text,content='chunks',content_rowid='id',tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
 INSERT INTO chunks_fts(rowid,title,text) VALUES(new.id,new.title,new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
 INSERT INTO chunks_fts(chunks_fts,rowid,title,text) VALUES('delete',old.id,old.title,old.text);
END;
CREATE TABLE IF NOT EXISTS sync_state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS budgets(key TEXT PRIMARY KEY,used INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS provider_health(provider TEXT PRIMARY KEY,retry_at INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS answer_cache(key TEXT PRIMARY KEY,response TEXT NOT NULL,expires INTEGER NOT NULL);
