import subprocess
import sys

def install(package):
    """function to install a package"""
    print(f"Installing the package {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def connect():
    """function to connect to mysql database"""
    import mysql.connector
    mydb = mysql.connector.connect(
      host=str(sys.argv[1]),
      user="root",
      password=str(sys.argv[2])
    )

    print(mydb)
    mycursor = mydb.cursor()

    mycursor.execute("CREATE DATABASE IF NOT EXISTS example")

    mycursor.execute("SHOW DATABASES")

    for x in mycursor:
        print(x)

# Example
if __name__ == '__main__':
    install('mysql-connector-python')
    connect()