import sqlite3
from .utils import *


class DBmanager:

	def __init__(self,con=None):

		self.now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
		self.con = con

		if not self.con:
			self.con = sqlite3.connect(f"model-{self.now}.db")
			self.con.executescript('./create_tables.sql')
			
	def insert(self,table,columns,items):
		cursor = self.con.cursor()
		self.columns = columns
		colstr = ','.join(self.columns)
		placeholders = ','.join(['?']*len(self.columns))
		cursor.executemany(f"""
							INSERT INTO {table} 
							({colstr}) 
							VALUES ({placeholders})""", 
					items)
		self.con.commit()

