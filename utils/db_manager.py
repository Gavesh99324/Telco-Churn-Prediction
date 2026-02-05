"""Database abstraction (RDS/SQLite)"""
import sqlite3
import os


class DBManager:
    """Database manager supporting SQLite and RDS"""

    def __init__(self, db_type='sqlite', db_name='churn.db'):
        self.db_type = db_type
        self.db_name = db_name
        self.conn = None

    def connect(self):
        """Connect to database"""
        if self.db_type == 'sqlite':
            self.conn = sqlite3.connect(self.db_name)
        return self.conn

    def execute_query(self, query):
        """Execute SQL query"""
        cursor = self.conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
