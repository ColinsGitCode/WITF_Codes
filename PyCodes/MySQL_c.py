#! python3
# Filename: Mysql_Class.py 

class MySQL:
    '''Class for MySql connections'''
    def __init__(self, Host="localhost", User="root", Password="", DB="Epinions"):
        self.Host = Host
        self.User = User
        self.Password = Password
        self.DB = DB
    
    def connect2MySql(self):
        self.ConnectDB = pymysql.connect(self.Host, self.User, self.Password, self.DB)
        
    def executeSQL(self, sql_str):
        # Execure SQL and return all results
        self.Cursor = self.ConnectDB.cursor()
        self.Cursor.execute(sql_str)
        return self.Cursor.fetchall()
