# ============================================================================
# Databases - Python Implementation
# ============================================================================

# Agenda
# Thanks to Prof. Cosma Shalizi (CMU Statistics) for this material
# - What databases are, and why
# - SQL
# - Interfacing Python and SQL

# ============================================================================
# What are Databases?
# ============================================================================

# A record is a collection of fields
# A table is a collection of records which all have the same fields (with different values)
# A database is a collection of tables

# ============================================================================
# Connecting Python to SQL Databases
# ============================================================================

# There are several packages for connecting Python to SQL databases:
# - sqlite3: for SQLite databases (built-in, no installation needed)
# - pymysql/mysql-connector-python: for MySQL/MariaDB databases
# - psycopg2: for PostgreSQL databases
# - sqlalchemy: general ORM and interface for various databases
# - pandas: has built-in SQL support via read_sql and to_sql

import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
import os

# ============================================================================
# Example 1: SQLite Database Connection (using sqlite3)
# ============================================================================

# Create a connection to SQLite database
# If file doesn't exist, it will be created
conn = sqlite3.connect('my_database.db')
cursor = conn.cursor()

# Create a sample table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        grade REAL
    )
""")

# Insert data into the table
cursor.execute("""
    INSERT INTO students (name, age, grade) VALUES 
    ('Alice', 20, 85.5),
    ('Bob', 22, 92.0),
    ('Charlie', 21, 78.5),
    ('Diana', 23, 88.0)
""")

# Commit the changes
conn.commit()

# Query data from the table
cursor.execute("SELECT * FROM students")
result = cursor.fetchall()
print("All students:")
print(result)

# Query with filtering
cursor.execute("SELECT name, grade FROM students WHERE grade > 85")
high_performers = cursor.fetchall()
print("\nHigh performers:")
print(high_performers)

# Query with aggregation
cursor.execute("SELECT AVG(grade) as average_grade FROM students")
avg_grade = cursor.fetchone()
print(f"\nAverage grade: {avg_grade[0]}")

# Close the connection when done
cursor.close()
conn.close()

# ============================================================================
# Example 2: Using pandas with SQLite
# ============================================================================

# Connect to the database
conn = sqlite3.connect('my_database.db')

# Read entire table into a DataFrame
students_df = pd.read_sql_query("SELECT * FROM students", conn)
print("\nStudents DataFrame:")
print(students_df)

# Query with filtering using pandas
high_performers_df = pd.read_sql_query(
    "SELECT name, grade FROM students WHERE grade > 85", 
    conn
)
print("\nHigh performers DataFrame:")
print(high_performers_df)

# Close connection
conn.close()

# ============================================================================
# Example 3: Using SQLAlchemy (Recommended for Complex Applications)
# ============================================================================

# Create an SQLAlchemy engine
engine = create_engine('sqlite:///my_database.db')

# Create a sample dataframe
students_data = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [20, 22, 21, 23, 20],
    'grade': [85.5, 92.0, 78.5, 88.0, 95.5]
})

# Write dataframe to database
students_data.to_sql('students', engine, if_exists='replace', index=False)

# Read entire table back
with engine.connect() as conn:
    students_from_db = pd.read_sql_table('students', conn)
    print("\nStudents from database:")
    print(students_from_db)

# Execute raw SQL with SQLAlchemy
with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM students WHERE age > 20"))
    print("\nStudents older than 20:")
    for row in result:
        print(row)

# Get list of tables
from sqlalchemy import inspect
inspector = inspect(engine)
tables = inspector.get_table_names()
print(f"\nTables in database: {tables}")

# ============================================================================
# Example 4: pandas DataFrame Operations with SQL
# ============================================================================

# Using pandas read_sql with more complex queries
with engine.connect() as conn:
    # Filtering and sorting
    query = """
        SELECT name, grade 
        FROM students 
        WHERE grade > 80 
        ORDER BY grade DESC
    """
    result_df = pd.read_sql_query(query, conn)
    print("\nFiltered and sorted:")
    print(result_df)

# ============================================================================
# Example 5: MySQL Database Connection
# ============================================================================

# For MySQL databases (requires MySQL server running)
# pip install pymysql

# import pymysql
# from sqlalchemy import create_engine
# 
# # Method 1: Using pymysql directly
# conn = pymysql.connect(
#     host='localhost',
#     port=3306,
#     user='username',
#     password='password',
#     database='my_database'
# )
# cursor = conn.cursor()
# cursor.execute("SELECT * FROM my_table")
# result = cursor.fetchall()
# conn.close()
# 
# # Method 2: Using SQLAlchemy
# engine = create_engine('mysql+pymysql://username:password@localhost:3306/my_database')
# df = pd.read_sql_query("SELECT * FROM my_table", engine)

# ============================================================================
# Example 6: PostgreSQL Database Connection
# ============================================================================

# For PostgreSQL databases (requires PostgreSQL server running)
# pip install psycopg2-binary

# import psycopg2
# from sqlalchemy import create_engine
# 
# # Method 1: Using psycopg2 directly
# conn = psycopg2.connect(
#     host='localhost',
#     port=5432,
#     user='username',
#     password='password',
#     database='my_database'
# )
# cursor = conn.cursor()
# cursor.execute("SELECT * FROM my_table")
# result = cursor.fetchall()
# conn.close()
# 
# # Method 2: Using SQLAlchemy
# engine = create_engine('postgresql://username:password@localhost:5432/my_database')
# df = pd.read_sql_query("SELECT * FROM my_table", engine)

# ============================================================================
# Common SQL Operations in Python
# ============================================================================

# Reconnect for examples
engine = create_engine('sqlite:///my_database.db')

# 1. SELECT - Retrieve data
with engine.connect() as conn:
    result = pd.read_sql_query("SELECT * FROM students", conn)
    print("\n1. SELECT all:")
    print(result)
    
    result = pd.read_sql_query("SELECT name, grade FROM students WHERE age > 20", conn)
    print("\n1. SELECT with WHERE:")
    print(result)

# 2. INSERT - Add new records
with engine.connect() as conn:
    conn.execute(text("INSERT INTO students (name, age, grade) VALUES ('Frank', 24, 89.0)"))
    conn.commit()
    print("\n2. INSERT: Added Frank")

# 3. UPDATE - Modify existing records
with engine.connect() as conn:
    conn.execute(text("UPDATE students SET grade = 90.0 WHERE name = 'Frank'"))
    conn.commit()
    print("\n3. UPDATE: Updated Frank's grade")

# 4. DELETE - Remove records
with engine.connect() as conn:
    conn.execute(text("DELETE FROM students WHERE name = 'Frank'"))
    conn.commit()
    print("\n4. DELETE: Removed Frank")

# 5. ORDER BY - Sort results
with engine.connect() as conn:
    result = pd.read_sql_query("SELECT * FROM students ORDER BY grade DESC", conn)
    print("\n5. ORDER BY grade DESC:")
    print(result)

# 6. GROUP BY - Aggregate data
with engine.connect() as conn:
    result = pd.read_sql_query(
        "SELECT age, AVG(grade) as avg_grade FROM students GROUP BY age", 
        conn
    )
    print("\n6. GROUP BY age:")
    print(result)

# 7. JOIN - Combine tables (example with two tables)
with engine.connect() as conn:
    # First create a courses table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS courses (
            student_id INTEGER,
            course_name TEXT,
            credits INTEGER
        )
    """))
    
    conn.execute(text("""
        INSERT INTO courses (student_id, course_name, credits) VALUES 
        (1, 'Math', 3),
        (1, 'Physics', 4),
        (2, 'Chemistry', 3),
        (3, 'Biology', 4)
    """))
    conn.commit()
    
    # Join students and courses
    result = pd.read_sql_query("""
        SELECT s.name, c.course_name, c.credits 
        FROM students s 
        JOIN courses c ON s.id = c.student_id
    """, conn)
    print("\n7. JOIN students and courses:")
    print(result)

# 8. COUNT, SUM, AVG, MIN, MAX
with engine.connect() as conn:
    result = pd.read_sql_query("""
        SELECT 
            COUNT(*) as total_students,
            AVG(grade) as avg_grade,
            MIN(grade) as min_grade,
            MAX(grade) as max_grade
        FROM students
    """, conn)
    print("\n8. Aggregate functions:")
    print(result)

# ============================================================================
# Using Parameterized Queries (Prevent SQL Injection)
# ============================================================================

with engine.connect() as conn:
    # Parameterized query using SQLAlchemy
    name_to_find = 'Alice'
    result = pd.read_sql_query(
        text("SELECT * FROM students WHERE name = :name"),
        conn,
        params={'name': name_to_find}
    )
    print("\nParameterized query result:")
    print(result)

# ============================================================================
# Using Context Managers and Transactions
# ============================================================================

# Using context manager for automatic resource cleanup
with engine.connect() as conn:
    with conn.begin():  # Start transaction
        conn.execute(text("INSERT INTO students (name, age, grade) VALUES ('Grace', 22, 87.0)"))
        conn.execute(text("INSERT INTO students (name, age, grade) VALUES ('Henry', 21, 91.5)"))
        # Automatically commits if no exception, rolls back if exception occurs

print("\nTransaction completed")

# ============================================================================
# Using pandasql for SQL on DataFrames (Alternative to SQL databases)
# ============================================================================

# pip install pandasql

# from pandasql import sqldf
# 
# # Create sample dataframes
# students_df = pd.DataFrame({
#     'name': ['Alice', 'Bob', 'Charlie'],
#     'age': [20, 22, 21],
#     'grade': [85.5, 92.0, 78.5]
# })
# 
# # Run SQL query on the dataframe
# query = "SELECT name, grade FROM students_df WHERE grade > 80"
# result = sqldf(query, locals())
# print(result)

# ============================================================================
# Best Practices
# ============================================================================

# 1. Always close connections or use context managers
#    with engine.connect() as conn:
#        # operations
#    # Connection automatically closed

# 2. Use parameterized queries to prevent SQL injection
#    conn.execute(text("SELECT * FROM students WHERE name = :name"), {'name': name_var})

# 3. Use transactions for multiple related operations
#    with conn.begin():
#        # multiple operations
#        # automatically commits or rolls back

# 4. Use pandas for easy data manipulation
#    df = pd.read_sql_query(query, conn)

# 5. Use SQLAlchemy for database portability
#    Easy to switch between SQLite, MySQL, PostgreSQL, etc.

# 6. Handle exceptions properly
#    try:
#        # database operations
#    except Exception as e:
#        conn.rollback()
#        print(f"Error: {e}")

# ============================================================================
# Cleanup
# ============================================================================

# Close engine
engine.dispose()

# Optionally, remove the database file
# os.remove('my_database.db')

print("\n" + "="*80)
print("Summary")
print("="*80)
print("""
A database is basically a way of dealing efficiently with lots of 
potentially huge dataframes

SQL is the standard language for telling databases what to do, 
especially what queries to run

Everything in an SQL query is something we've practiced already in Python:
- Filtering (WHERE clause = boolean indexing)
- Aggregation (GROUP BY = groupby())
- Merging (JOIN = merge())
- Ordering (ORDER BY = sort_values())

Workflow:
1. Connect Python to the database
2. Send it an SQL query
3. Analyze the returned DataFrame

More information at: http://www.stat.cmu.edu/~cshalizi/statcomp/14/
""")
