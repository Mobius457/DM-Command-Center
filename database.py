import sqlite3

def init_db(db_path):
    """
    Initializes a new SQLite database with the required schema.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # --- Create NPCs Table ---
        # Stores generated non-player characters
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS npcs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                portrait BLOB
            );
        """)

        # --- Create Locations Table ---
        # Stores generated locations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT
            );
        """)

        # --- Create Maps Table ---
        # Stores generated maps
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS maps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                map_image BLOB
            );
        """)
        
        # --- Create Quests Table ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                outline TEXT
            );
        """)

        conn.commit()
        print(f"Database initialized successfully at {db_path}")

    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()

def add_npc(db_path, name, description, portrait_blob=None):
    """Adds a new NPC to the database."""
    if not db_path:
        return
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO npcs (name, description, portrait) VALUES (?, ?, ?)",
            (name, description, portrait_blob)
        )
        conn.commit()
        print(f"NPC '{name}' added to the database.")
    except sqlite3.Error as e:
        print(f"Database error adding NPC: {e}")
    finally:
        if conn:
            conn.close()

def get_all_npcs(db_path):
    """Retrieves all NPCs (id, name) from the database."""
    if not db_path:
        return []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM npcs ORDER BY name ASC")
        npcs = cursor.fetchall()
        return npcs
    except sqlite3.Error as e:
        print(f"Database error getting NPCs: {e}")
        return []
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # For testing purposes, create a dummy database if the module is run directly
    init_db("test_campaign.db")
