# ============================================================================
# Databases
# ============================================================================

# Agenda
# Thanks to Prof. Cosma Shalizi (CMU Statistics) for this material
# - What databases are, and why
# - SQL
# - Interfacing R and SQL

# ============================================================================
# What are Databases?
# ============================================================================

# A record is a collection of fields
# A table is a collection of records which all have the same fields (with different values)
# A database is a collection of tables

# ============================================================================
# Connecting R to SQL Databases
# ============================================================================

# There are several packages for connecting R to SQL databases:
# - RSQLite: for SQLite databases (file-based, no server needed)
# - RMySQL: for MySQL/MariaDB databases
# - RPostgreSQL: for PostgreSQL databases
# - odbc/DBI: general interface for various databases

# ============================================================================
# Example 1: SQLite Database Connection
# ============================================================================

# Install and load required packages
# install.packages("RSQLite")
# install.packages("DBI")

library(DBI)
library(RSQLite)

# Create a connection to SQLite database
# If file doesn't exist, it will be created
con <- dbConnect(RSQLite::SQLite(), "my_database.db")

# Create a sample table
dbExecute(con, "CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    grade REAL
)")

# Insert data into the table
dbExecute(con, "INSERT INTO students (name, age, grade) VALUES 
    ('Alice', 20, 85.5),
    ('Bob', 22, 92.0),
    ('Charlie', 21, 78.5),
    ('Diana', 23, 88.0)")

# Query data from the table
result <- dbGetQuery(con, "SELECT * FROM students")
print(result)

# Query with filtering
high_performers <- dbGetQuery(con, "SELECT name, grade FROM students WHERE grade > 85")
print(high_performers)

# Query with aggregation
avg_grade <- dbGetQuery(con, "SELECT AVG(grade) as average_grade FROM students")
print(avg_grade)

# Close the connection when done
dbDisconnect(con)

# ============================================================================
# Example 2: Using dbplyr for dplyr-style Queries
# ============================================================================

# install.packages("dbplyr")
library(dbplyr)
library(dplyr)

# Reconnect to the database
con <- dbConnect(RSQLite::SQLite(), "my_database.db")

# Create a table reference (lazy evaluation)
students_tbl <- tbl(con, "students")

# Use dplyr verbs - these generate SQL behind the scenes
students_tbl %>%
  filter(grade > 80) %>%
  select(name, grade) %>%
  arrange(desc(grade)) %>%
  collect()  # collect() executes the query and brings data into R

# Show the SQL that would be generated
students_tbl %>%
  filter(grade > 80) %>%
  select(name, grade) %>%
  show_query()

# Close connection
dbDisconnect(con)

# ============================================================================
# Example 3: Working with Existing Data Frames
# ============================================================================

# Create a sample dataframe
students_df <- data.frame(
  id = 1:5,
  name = c("Alice", "Bob", "Charlie", "Diana", "Eve"),
  age = c(20, 22, 21, 23, 20),
  grade = c(85.5, 92.0, 78.5, 88.0, 95.5),
  stringsAsFactors = FALSE
)

# Connect to database
con <- dbConnect(RSQLite::SQLite(), "my_database.db")

# Write dataframe to database
dbWriteTable(con, "students", students_df, overwrite = TRUE)

# Read entire table back
students_from_db <- dbReadTable(con, "students")
print(students_from_db)

# List all tables in the database
dbListTables(con)

# Get column names for a table
dbListFields(con, "students")

# Close connection
dbDisconnect(con)

# ============================================================================
# Example 4: MySQL Database Connection
# ============================================================================

# For MySQL databases (requires MySQL server running)
# install.packages("RMySQL")

# library(RMySQL)
# 
# # Connect to MySQL database
# con <- dbConnect(RMySQL::MySQL(),
#                  host = "localhost",
#                  port = 3306,
#                  dbname = "my_database",
#                  user = "username",
#                  password = "password")
# 
# # Run queries
# result <- dbGetQuery(con, "SELECT * FROM my_table")
# 
# # Close connection
# dbDisconnect(con)

# ============================================================================
# Example 5: PostgreSQL Database Connection
# ============================================================================

# For PostgreSQL databases (requires PostgreSQL server running)
# install.packages("RPostgres")

# library(DBI)
# library(RPostgres)
# 
# # Connect to PostgreSQL database
# con <- dbConnect(RPostgres::Postgres(),
#                  host = "localhost",
#                  port = 5432,
#                  dbname = "my_database",
#                  user = "username",
#                  password = "password")
# 
# # Run queries
# result <- dbGetQuery(con, "SELECT * FROM my_table")
# 
# # Close connection
# dbDisconnect(con)

# ============================================================================
# Common SQL Operations in R
# ============================================================================

# Connect to database for examples
con <- dbConnect(RSQLite::SQLite(), "my_database.db")

# 1. SELECT - Retrieve data
dbGetQuery(con, "SELECT * FROM students")
dbGetQuery(con, "SELECT name, grade FROM students WHERE age > 20")

# 2. INSERT - Add new records
dbExecute(con, "INSERT INTO students (name, age, grade) VALUES ('Frank', 24, 89.0)")

# 3. UPDATE - Modify existing records
dbExecute(con, "UPDATE students SET grade = 90.0 WHERE name = 'Frank'")

# 4. DELETE - Remove records
dbExecute(con, "DELETE FROM students WHERE name = 'Frank'")

# 5. ORDER BY - Sort results
dbGetQuery(con, "SELECT * FROM students ORDER BY grade DESC")

# 6. GROUP BY - Aggregate data
dbGetQuery(con, "SELECT age, AVG(grade) as avg_grade FROM students GROUP BY age")

# 7. JOIN - Combine tables (example with two tables)
# First create a courses table
dbExecute(con, "CREATE TABLE IF NOT EXISTS courses (
    student_id INTEGER,
    course_name TEXT,
    credits INTEGER
)")

dbExecute(con, "INSERT INTO courses (student_id, course_name, credits) VALUES 
    (1, 'Math', 3),
    (1, 'Physics', 4),
    (2, 'Chemistry', 3),
    (3, 'Biology', 4)")

# Join students and courses
dbGetQuery(con, "SELECT s.name, c.course_name, c.credits 
                 FROM students s 
                 JOIN courses c ON s.id = c.student_id")

# 8. COUNT, SUM, AVG, MIN, MAX
dbGetQuery(con, "SELECT 
    COUNT(*) as total_students,
    AVG(grade) as avg_grade,
    MIN(grade) as min_grade,
    MAX(grade) as max_grade
    FROM students")

# Close connection
dbDisconnect(con)

# ============================================================================
# Best Practices
# ============================================================================

# 1. Always close connections when done
#    Use dbDisconnect(con)

# 2. Use parameterized queries to prevent SQL injection
#    dbGetQuery(con, "SELECT * FROM students WHERE name = ?", params = list(name_var))

# 3. Use transactions for multiple related operations
#    dbBegin(con)
#    # ... multiple operations ...
#    dbCommit(con)  # or dbRollback(con) if error

# 4. Check if table exists before creating
#    dbExistsTable(con, "table_name")

# 5. Use dbplyr for complex data manipulation
#    It translates dplyr code to SQL automatically

# ============================================================================
# Going the Other Way: Using sqldf Package
# ============================================================================

# The sqldf package lets you use SQL commands on dataframes
# Mostly useful if you already know SQL better than R…

# Example usage:
# library(sqldf)
# sqldf("SELECT * FROM my_dataframe WHERE column > 10")

# ============================================================================
# Summary
# ============================================================================

# A database is basically a way of dealing efficiently with lots of 
# potentially huge dataframes

# SQL is the standard language for telling databases what to do, 
# especially what queries to run

# Everything in an SQL query is something we've practiced already in R:
# - subsetting/selection
# - aggregation
# - merging
# - ordering

# Workflow:
# 1. Connect R to the database
# 2. Send it an SQL query
# 3. Analyse the returned dataframe

# More information at: http://www.stat.cmu.edu/~cshalizi/statcomp/14/
