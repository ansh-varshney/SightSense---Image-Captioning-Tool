import sqlite3

def setup_database():
    conn = sqlite3.connect('C:/Users/Ansh Varshney/Desktop/Image_Captioning/backend/database/app.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS captions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            caption TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_to_database(image_path, caption):
    conn = sqlite3.connect('C:/Users/Ansh Varshney/Desktop/Image_Captioning/backend/database/app.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO captions (image_path, caption) VALUES (?, ?)', (image_path, caption))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    setup_database()
