import sqlite3
import sys

db_path = sys.argv[1]
conn = sqlite3.connect(db_path)

print("\n=== EXTRACTED ARTICLES ===")
cursor = conn.execute("SELECT bibliographic__title, bibliographic__publication_year FROM articles LIMIT 3")
for i, row in enumerate(cursor.fetchall(), 1):
    print(f"{i}. {row[0][:80]}... ({row[1]})")

print("\n=== ALGORITHMS USED ===")
cursor = conn.execute("SELECT algorithm, COUNT(*) as cnt FROM article_algorithms GROUP BY algorithm")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} studies")

print("\n=== XAI METHODS ===")
cursor = conn.execute("SELECT method, COUNT(*) as cnt FROM article_xai_methods GROUP BY method")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} studies")

conn.close()
