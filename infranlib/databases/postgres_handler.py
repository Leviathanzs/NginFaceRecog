import os
import psycopg2
import time
import threading
from psycopg2 import pool
from psycopg2 import Error
from dotenv import load_dotenv
import os

load_dotenv(os.path.dirname(os.path.abspath(__file__))+'/.env')

print(os.getenv('DB_USERNAME'))

class DatabaseHeartBeat(threading.Thread):
    def __init__(self, dbh, logging):
        threading.Thread.__init__(self)
        self.dbh = dbh
        self.logging = logging
    
    def run(self):
        while True:
            # self.logging.info("Heart Beat Postgresql.........")
            self.dbh.execute('SELECT 1')
            time.sleep(10)

class DatabaseHandler():

    def __init__(self):
        self.connectionPool = None
        self.max_try = 5
        try:
            # Connect to an existing database
            self.connectionPool = psycopg2.pool.ThreadedConnectionPool(1,10,
                                        user=os.getenv('DB_USERNAME'),
                                        password=os.getenv('DB_PASSWORD'),
                                        host=os.getenv('DB_HOST'),
                                        port=os.getenv('DB_PORT'),
                                        database=os.getenv('DB_DATABASE'))
            if self.connectionPool:
                # print("Connection pool created successfully")
                pg_conn = self.connectionPool.getconn()
                # Create a cursor to perform database operations
                if pg_conn:
                    cursor = pg_conn.cursor()
                    # Print PostgreSQL details
                    # print("PostgreSQL server information")
                    # print(self.connection.get_dsn_parameters(), "\n")
                    # Executing a SQL query
                    cursor.execute("SELECT version();")
                    # Fetch result
                    record = cursor.fetchone()
                    print("Connected to - ", record, "\n")
                else:
                    print("Failed to get connection from pool")

        except (Exception, Error) as error:
            print("Error while connecting to PostgreSQL", error)
        finally:
            if (self.connectionPool):
                # cursor.close()
                # self.connectionPool.closeall()
                print("Connection pool created successfully")

    def get_connection(self):
        return self.connectionPool.getconn()
    
    def close_connection(self, connection):
        self.connectionPool.putconn(connection)
    
    def close_all_connection(self):
        self.connectionPool.closeall()
    
    def fetchone(self, query, params=None, try_count=0):
        if try_count<self.max_try:
            try:
                connection = self.get_connection()
                cursor = connection.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                result = cursor.fetchone()
                self.close_connection(connection)
            except psycopg2.OperationalError:
                print("OperationalError when execute, try again (%i/%i)"%(try_count+1, self.max_try))
                self.fetchone(query, params, try_count=try_count+1)
        else:
            print("Max try exceed please restart driver")

        return result
    
    def fetchall(self, query, params=None, try_count=0):
        if try_count<self.max_try:
            try:
                connection = self.get_connection()
                cursor = connection.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                result = cursor.fetchall()
                self.close_connection(connection)
            except psycopg2.OperationalError:
                print("OperationalError when execute, try again (%i/%i)"%(try_count+1, self.max_try))
                self.fetchall(query, params, try_count=try_count+1)
        else:
            print("Max try exceed please restart driver")
        
        return result
    
    def fetchmany(self, query, params=None, size=1, try_count=0):
        if try_count<self.max_try:
            try:
                connection = self.get_connection()
                cursor = connection.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                result = cursor.fetchmany(size)
                self.close_connection(connection)
            except psycopg2.OperationalError:
                print("OperationalError when execute, try again (%i/%i)"%(try_count+1, self.max_try))
                self.fetchmany(query, params, size, try_count=try_count+1)
        else:
            print("Max try exceed please restart driver")
        
        return result
    
    def execute(self, query, params=None, commit=True, fetch=False, try_count=0):
        if try_count<self.max_try:
            try:
                connection = self.get_connection()
                cursor = connection.cursor()
                result = None
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                if commit:
                    connection.commit()
                if fetch:
                    result = cursor.fetchall()
                # print(result)
                self.close_connection(connection)
            except psycopg2.OperationalError:
                print("OperationalError when execute, try again (%i/%i)"%(try_count+1, self.max_try))
                self.execute(query, params, commit, fetch, try_count=try_count+1)
        else:
            print("Max try exceed please restart driver")
        
        return result
    
    def commit(self):
        connection = self.get_connection()
        connection.commit()
        self.close_connection(connection)
    
    def __del__(self):
        if (self.connectionPool):
            self.connectionPool.closeall()
            print("Connection pool closed successfully")

if __name__ == '__main__':
    dbh = DatabaseHandler()

