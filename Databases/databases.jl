# ============================================================================
# Databases - Julia Implementation
# ============================================================================

# Agenda
# Thanks to Prof. Cosma Shalizi (CMU Statistics) for this material
# - What databases are, and why
# - SQL
# - Interfacing Julia and SQL

# ============================================================================
# What are Databases?
# ============================================================================

# A record is a collection of fields
# A table is a collection of records which all have the same fields (with different values)
# A database is a collection of tables

# ============================================================================
# Connecting Julia to SQL Databases
# ============================================================================

# There are several packages for connecting Julia to SQL databases:
# - SQLite.jl: for SQLite databases (file-based, no server needed)
# - MySQL.jl: for MySQL/MariaDB databases
# - LibPQ.jl: for PostgreSQL databases
# - DBInterface.jl: general interface for various databases
# - DataFrames.jl: for working with tabular data

using SQLite
using DataFrames
using DBInterface

# ============================================================================
# Example 1: SQLite Database Connection
# ============================================================================

# Create a connection to SQLite database
# If file doesn't exist, it will be created
db = SQLite.DB("my_database.db")

# Create a sample table
DBInterface.execute(db, """
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        grade REAL
    )
""")

# Insert data into the table
DBInterface.execute(db, """
    INSERT INTO students (name, age, grade) VALUES 
    ('Alice', 20, 85.5),
    ('Bob', 22, 92.0),
    ('Charlie', 21, 78.5),
    ('Diana', 23, 88.0)
""")

# Query data from the table
result = DBInterface.execute(db, "SELECT * FROM students") |> DataFrame
println("All students:")
println(result)

# Query with filtering
high_performers = DBInterface.execute(db, 
    "SELECT name, grade FROM students WHERE grade > 85") |> DataFrame
println("\nHigh performers:")
println(high_performers)

# Query with aggregation
avg_grade = DBInterface.execute(db, 
    "SELECT AVG(grade) as average_grade FROM students") |> DataFrame
println("\nAverage grade:")
println(avg_grade)

# ============================================================================
# Example 2: Using SQLite.jl Directly
# ============================================================================

# Query returns a DataFrame directly
students_df = SQLite.Query(db, "SELECT * FROM students") |> DataFrame
println("\nStudents DataFrame:")
println(students_df)

# Query with filtering
high_performers_df = SQLite.Query(db, 
    "SELECT name, grade FROM students WHERE grade > 85") |> DataFrame
println("\nHigh performers DataFrame:")
println(high_performers_df)

# ============================================================================
# Example 3: Working with Existing DataFrames
# ============================================================================

# Create a sample DataFrame
students_data = DataFrame(
    id = [1, 2, 3, 4, 5],
    name = ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    age = [20, 22, 21, 23, 20],
    grade = [85.5, 92.0, 78.5, 88.0, 95.5]
)

# Write DataFrame to database
SQLite.load!(students_data, db, "students", replace=true)

# Read entire table back
students_from_db = SQLite.Query(db, "SELECT * FROM students") |> DataFrame
println("\nStudents from database:")
println(students_from_db)

# Get list of tables in the database
tables = SQLite.tables(db) |> DataFrame
println("\nTables in database:")
println(tables)

# Get column information for a table
columns = SQLite.columns(db, "students") |> DataFrame
println("\nColumns in students table:")
println(columns)

# ============================================================================
# Example 4: MySQL Database Connection
# ============================================================================

# For MySQL databases (requires MySQL server running)
# using MySQL
# 
# # Connect to MySQL database
# conn = MySQL.connect(
#     host="localhost",
#     port=3306,
#     user="username",
#     passwd="password",
#     db="my_database"
# )
# 
# # Run queries
# result = MySQL.query(conn, "SELECT * FROM my_table") |> DataFrame
# 
# # Close connection
# MySQL.disconnect(conn)

# ============================================================================
# Example 5: PostgreSQL Database Connection
# ============================================================================

# For PostgreSQL databases (requires PostgreSQL server running)
# using LibPQ
# 
# # Connect to PostgreSQL database
# conn = LibPQ.Connection("""
#     host=localhost
#     port=5432
#     dbname=my_database
#     user=username
#     password=password
# """)
# 
# # Run queries
# result = LibPQ.execute(conn, "SELECT * FROM my_table") |> DataFrame
# 
# # Close connection
# close(conn)

# ============================================================================
# Common SQL Operations in Julia
# ============================================================================

# 1. SELECT - Retrieve data
result = DBInterface.execute(db, "SELECT * FROM students") |> DataFrame
println("\n1. SELECT all:")
println(result)

result = DBInterface.execute(db, 
    "SELECT name, grade FROM students WHERE age > 20") |> DataFrame
println("\n1. SELECT with WHERE:")
println(result)

# 2. INSERT - Add new records
DBInterface.execute(db, 
    "INSERT INTO students (name, age, grade) VALUES ('Frank', 24, 89.0)")
println("\n2. INSERT: Added Frank")

# 3. UPDATE - Modify existing records
DBInterface.execute(db, 
    "UPDATE students SET grade = 90.0 WHERE name = 'Frank'")
println("\n3. UPDATE: Updated Frank's grade")

# 4. DELETE - Remove records
DBInterface.execute(db, 
    "DELETE FROM students WHERE name = 'Frank'")
println("\n4. DELETE: Removed Frank")

# 5. ORDER BY - Sort results
result = DBInterface.execute(db, 
    "SELECT * FROM students ORDER BY grade DESC") |> DataFrame
println("\n5. ORDER BY grade DESC:")
println(result)

# 6. GROUP BY - Aggregate data
result = DBInterface.execute(db, 
    "SELECT age, AVG(grade) as avg_grade FROM students GROUP BY age") |> DataFrame
println("\n6. GROUP BY age:")
println(result)

# 7. JOIN - Combine tables (example with two tables)
# First create a courses table
DBInterface.execute(db, """
    CREATE TABLE IF NOT EXISTS courses (
        student_id INTEGER,
        course_name TEXT,
        credits INTEGER
    )
""")

DBInterface.execute(db, """
    INSERT INTO courses (student_id, course_name, credits) VALUES 
    (1, 'Math', 3),
    (1, 'Physics', 4),
    (2, 'Chemistry', 3),
    (3, 'Biology', 4)
""")

# Join students and courses
result = DBInterface.execute(db, """
    SELECT s.name, c.course_name, c.credits 
    FROM students s 
    JOIN courses c ON s.id = c.student_id
""") |> DataFrame
println("\n7. JOIN students and courses:")
println(result)

# 8. COUNT, SUM, AVG, MIN, MAX
result = DBInterface.execute(db, """
    SELECT 
        COUNT(*) as total_students,
        AVG(grade) as avg_grade,
        MIN(grade) as min_grade,
        MAX(grade) as max_grade
    FROM students
""") |> DataFrame
println("\n8. Aggregate functions:")
println(result)

# ============================================================================
# Using Parameterized Queries (Prevent SQL Injection)
# ============================================================================

# SQLite.jl supports parameterized queries
name_to_find = "Alice"
stmt = SQLite.Stmt(db, "SELECT * FROM students WHERE name = ?")
DBInterface.execute(stmt, [name_to_find])
result = DataFrame(stmt)
println("\nParameterized query result:")
println(result)

# Alternative using bind parameters
result = DBInterface.execute(db, 
    "SELECT * FROM students WHERE name = ?", [name_to_find]) |> DataFrame
println("\nParameterized query (alternative):")
println(result)

# ============================================================================
# Using Prepared Statements for Better Performance
# ============================================================================

# Prepare statement once, execute multiple times
stmt = SQLite.Stmt(db, "INSERT INTO students (name, age, grade) VALUES (?, ?, ?)")

# Execute with different parameters
DBInterface.execute(stmt, ["Grace", 22, 87.0])
DBInterface.execute(stmt, ["Henry", 21, 91.5])

println("\nInserted multiple rows using prepared statement")

# Verify the insertions
result = DBInterface.execute(db, 
    "SELECT * FROM students WHERE name IN ('Grace', 'Henry')") |> DataFrame
println(result)

# ============================================================================
# Transactions
# ============================================================================

# Begin a transaction
SQLite.transaction(db) do
    DBInterface.execute(db, 
        "INSERT INTO students (name, age, grade) VALUES ('Isabel', 20, 88.5)")
    DBInterface.execute(db, 
        "INSERT INTO students (name, age, grade) VALUES ('Jack', 23, 93.0)")
    # Automatically commits if no exception, rolls back if exception occurs
end

println("\nTransaction completed")

# ============================================================================
# Working with DataFrames and SQL
# ============================================================================

using Statistics

# Query to DataFrame
students_df = SQLite.Query(db, "SELECT * FROM students") |> DataFrame

# Use DataFrames operations (alternative to SQL)
# Filter
high_performers = filter(row -> row.grade > 85, students_df)
println("\nHigh performers (using DataFrames):")
println(high_performers)

# Sort
sorted_students = sort(students_df, :grade, rev=true)
println("\nSorted by grade (using DataFrames):")
println(sorted_students)

# Group by and aggregate
using DataFramesMeta
grouped = @chain students_df begin
    groupby(:age)
    @combine(:avg_grade = mean(:grade))
end
println("\nGrouped by age (using DataFrames):")
println(grouped)

# ============================================================================
# Query Builder Pattern
# ============================================================================

# Building queries programmatically
function build_query(table::String; 
                     columns::Vector{String}=["*"], 
                     where::Union{String,Nothing}=nothing,
                     order_by::Union{String,Nothing}=nothing)
    query = "SELECT $(join(columns, ", ")) FROM $table"
    
    if !isnothing(where)
        query *= " WHERE $where"
    end
    
    if !isnothing(order_by)
        query *= " ORDER BY $order_by"
    end
    
    return query
end

# Use the query builder
query = build_query("students", 
                   columns=["name", "grade"], 
                   where="grade > 85",
                   order_by="grade DESC")
println("\nBuilt query: $query")
result = DBInterface.execute(db, query) |> DataFrame
println(result)

# ============================================================================
# Handling NULL Values
# ============================================================================

# Insert a record with NULL
DBInterface.execute(db, 
    "INSERT INTO students (name, age, grade) VALUES ('Karen', NULL, 85.0)")

# Query with NULL handling
result = DBInterface.execute(db, 
    "SELECT name, age, grade FROM students WHERE age IS NULL") |> DataFrame
println("\nStudents with NULL age:")
println(result)

# Clean up
DBInterface.execute(db, "DELETE FROM students WHERE name = 'Karen'")

# ============================================================================
# Batch Operations
# ============================================================================

# Insert multiple rows efficiently
new_students = DataFrame(
    name = ["Liam", "Mia", "Noah"],
    age = [22, 21, 23],
    grade = [88.5, 91.0, 87.5]
)

SQLite.load!(new_students, db, "students")
println("\nBatch insert completed")

# ============================================================================
# Database Inspection
# ============================================================================

# Get database schema
schema = SQLite.Query(db, "SELECT sql FROM sqlite_master WHERE type='table'") |> DataFrame
println("\nDatabase schema:")
println(schema)

# Get table info
table_info = SQLite.Query(db, "PRAGMA table_info(students)") |> DataFrame
println("\nTable info for students:")
println(table_info)

# Get indices
indices = SQLite.Query(db, "PRAGMA index_list(students)") |> DataFrame
println("\nIndices for students:")
println(indices)

# ============================================================================
# Best Practices
# ============================================================================

# 1. Use parameterized queries to prevent SQL injection
#    DBInterface.execute(db, "SELECT * FROM students WHERE name = ?", [name])

# 2. Use transactions for multiple related operations
#    SQLite.transaction(db) do
#        # multiple operations
#    end

# 3. Use prepared statements for repeated queries
#    stmt = SQLite.Stmt(db, "SELECT * FROM students WHERE age = ?")
#    DBInterface.execute(stmt, [age_value])

# 4. Use DataFrames for easy data manipulation
#    df = SQLite.Query(db, query) |> DataFrame

# 5. Handle NULL values explicitly
#    Check for missing values in DataFrames

# 6. Close database connections when done
#    SQLite.DB connections are automatically managed in Julia

# 7. Use batch operations for better performance
#    SQLite.load! for bulk inserts

# ============================================================================
# Performance Tips
# ============================================================================

# 1. Create indices for frequently queried columns
DBInterface.execute(db, "CREATE INDEX IF NOT EXISTS idx_grade ON students(grade)")

# 2. Use ANALYZE to update statistics
DBInterface.execute(db, "ANALYZE")

# 3. Use EXPLAIN QUERY PLAN to understand query performance
explain = SQLite.Query(db, "EXPLAIN QUERY PLAN SELECT * FROM students WHERE grade > 85") |> DataFrame
println("\nQuery plan:")
println(explain)

# 4. Use bulk operations instead of row-by-row
#    SQLite.load! is much faster than individual inserts

# ============================================================================
# Cleanup
# ============================================================================

# SQLite connections in Julia are automatically cleaned up
# But you can explicitly close if needed
# SQLite.close(db)

# Optionally, remove the database file
# rm("my_database.db")

println("\n" * "="^80)
println("Summary")
println("="^80)
println("""
A database is basically a way of dealing efficiently with lots of 
potentially huge DataFrames

SQL is the standard language for telling databases what to do, 
especially what queries to run

Everything in an SQL query is something we've practiced already in Julia:
- Filtering (WHERE clause = filter())
- Aggregation (GROUP BY = groupby())
- Merging (JOIN = join())
- Ordering (ORDER BY = sort())

Workflow:
1. Connect Julia to the database
2. Send it an SQL query
3. Analyze the returned DataFrame

Key packages:
- SQLite.jl: SQLite databases
- MySQL.jl: MySQL databases
- LibPQ.jl: PostgreSQL databases
- DBInterface.jl: Common interface
- DataFrames.jl: Tabular data manipulation

More information at: http://www.stat.cmu.edu/~cshalizi/statcomp/14/
""")
