
import mysql.connector
import pandas as pd

# Connect to the database
my_db = mysql.connector.connect(host="localhost", user="root", passwd="naruto", database="test")
mycursor = my_db.cursor()

# Execute the query
mycursor.execute('SELECT * FROM test_results ')
z = mycursor.fetchall()

# Get column names
column_names = [i[0] for i in mycursor.description]

# Create a DataFrame
sql = pd.DataFrame(z, columns=column_names)

# Print the DataFrame
print(sql)
