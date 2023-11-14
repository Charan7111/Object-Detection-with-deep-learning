import sqlite3
import datetime

# Connect to the SQLite database
conn = sqlite3.connect('expenses.db')
c = conn.cursor()

# Create table if not exists
c.execute('''
    CREATE TABLE IF NOT EXISTS expenses (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Date TEXT,
        Category TEXT,
        Amount REAL,
        Description TEXT
    )
''')

# Function to add an expense
def add_expense(category, amount, description=''):
    date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('INSERT INTO expenses (Date, Category, Amount, Description) VALUES (?, ?, ?, ?)',
              (date, category, amount, description))
    conn.commit()

# Function to generate monthly report
def generate_monthly_report(month, year):
    c.execute('SELECT * FROM expenses WHERE strftime("%m", Date) = ? AND strftime("%Y", Date) = ?', (month, year))
    expenses = c.fetchall()

    total_expenses = sum(expense[3] for expense in expenses)

    print(f'Monthly Report for {month}-{year}:')
    for expense in expenses:
        print(f"{expense[1]} - {expense[2]}: ${expense[3]} - {expense[4]}")

    print(f'Total Expenses: ${total_expenses}')

# Example usage
add_expense('Groceries', 50.0, 'Weekly shopping')
add_expense('Rent', 1200.0, 'Month')
add_expense('Dining out', 100.0, 'Month')
generate_monthly_report('11', '2023')

# Close the connection
conn.close()