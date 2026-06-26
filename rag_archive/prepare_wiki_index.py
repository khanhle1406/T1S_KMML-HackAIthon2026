import sqlite3
import json
import re
import time
import os
from datasets import load_dataset

DB_PATH = "wiki_index.db"
CHUNK_SIZE = 700

def chunk_text(text, max_len=CHUNK_SIZE):
    """Chunks text into paragraphs or groups of sentences up to max_len characters."""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_len = 0
    
    for p in paragraphs:
        # If a single paragraph is very long, split it by sentence
        if len(p) > max_len:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            
            sentences = re.split(r'(?<=[.!?])\s+', p)
            curr_sent = []
            curr_sent_len = 0
            for s in sentences:
                if curr_sent_len + len(s) <= max_len:
                    curr_sent.append(s)
                    curr_sent_len += len(s) + 1
                else:
                    if curr_sent:
                        chunks.append(" ".join(curr_sent))
                    curr_sent = [s]
                    curr_sent_len = len(s)
            if curr_sent:
                chunks.append(" ".join(curr_sent))
        else:
            if current_len + len(p) <= max_len:
                current_chunk.append(p)
                current_len += len(p) + 1
            else:
                chunks.append("\n".join(current_chunk))
                current_chunk = [p]
                current_len = len(p)
                
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

def main():
    print("📥 Loading tdtunlp/wikipedia_vi dataset from Hugging Face...")
    t_start = time.time()
    dataset = load_dataset("tdtunlp/wikipedia_vi", split="train")
    print(f"✅ Loaded {len(dataset)} articles in {time.time() - t_start:.2f}s")
    
    # Remove existing db if any
    if os.path.exists(DB_PATH):
        print(f"🗑️ Removing existing database: {DB_PATH}")
        os.remove(DB_PATH)
        
    print(f"🛠️ Creating SQLite FTS5 database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create virtual table for full text search
    # We use the unicode61 tokenizer which supports accents and case folding
    cursor.execute("""
        CREATE VIRTUAL TABLE wiki_fts USING fts5(
            title, 
            text, 
            url UNINDEXED, 
            tokenize='unicode61'
        );
    """)
    conn.commit()
    
    print("🚀 Chunking articles and inserting into SQLite index...")
    batch_size = 50000
    insert_data = []
    
    t_insert_start = time.time()
    total_chunks = 0
    
    # We process in transactions for maximum speed
    cursor.execute("PRAGMA journal_mode = OFF;")
    cursor.execute("PRAGMA synchronous = OFF;")
    
    for idx, article in enumerate(dataset):
        title = article.get("title", "").strip()
        text = article.get("text", "").strip()
        url = article.get("url", "").strip()
        
        # Skip empty stubs or Main Page
        if not text or title == "Trang Chính" or len(text) < 40:
            continue
            
        chunks = chunk_text(text)
        for chunk in chunks:
            if len(chunk) < 30:
                continue
            insert_data.append((title, chunk, url))
            total_chunks += 1
            
        if len(insert_data) >= batch_size:
            cursor.execute("BEGIN TRANSACTION;")
            cursor.executemany("INSERT INTO wiki_fts (title, text, url) VALUES (?, ?, ?);", insert_data)
            conn.commit()
            insert_data = []
            print(f"💾 Indexed {idx+1}/{len(dataset)} articles ({total_chunks} chunks)...")
            
    if insert_data:
        cursor.execute("BEGIN TRANSACTION;")
        cursor.executemany("INSERT INTO wiki_fts (title, text, url) VALUES (?, ?, ?);", insert_data)
        conn.commit()
        
    duration = time.time() - t_insert_start
    print(f"✅ Successfully built Wikipedia FTS5 index!")
    print(f"📊 Total Articles: {len(dataset)}")
    print(f"📊 Total Chunks: {total_chunks}")
    print(f"⏱️ Time taken: {duration:.2f}s ({total_chunks / duration:.2f} chunks/sec)")
    
    # Print database size
    db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"📂 Database file size: {db_size_mb:.2f} MB")
    
    # Close connection
    conn.close()

if __name__ == "__main__":
    main()
