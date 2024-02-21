import sqlite3
from .utils import *


def get_class_def(self,loc):
	return {'id':id(self),'type':str(self),**loc}

class SQL:

    create_table = open(get_path(['create_tables.sql'])).read()
    
    neuron = lambda obj : [ 'neurons',
                    ['neuron_id','layer_id','type','n_in'],
                    [(obj.id['id'],obj.id['layer']['id'],obj.id['type'],obj.id['n_in'])]
                    ]

    layers = lambda obj : ['layers',
                            ['layer_id','type','n_out'],
                            [(obj.id['id'],obj.id['type'],obj.id['n_out'])]]
    weights = lambda obj : ['weights',
                            ['neuron_id','weight_id','value'],
                            [(obj.id['id'],ix,w) for ix,w in enumerate(obj.w.ravel())]]
	
class DBmanager:


	def __init__(self):
		...
	def start(self,db=None):
		self.db_path = db or get_path(['run',f"model{now()}.db"])
		DBmanager.con = sqlite3.connect(self.db_path)
		DBmanager.con.executescript(SQL.create_table)
		return self.db_path
			
	def insert_db(self,table,columns,items):
		cursor = DBmanager.con.cursor()
		self.columns = columns
		colstr = ','.join(self.columns)
		placeholders = ','.join(['?']*len(self.columns))
		query = f"""
					INSERT INTO {table} 
					({colstr}) 
					VALUES ({placeholders})
				"""
		cursor.executemany(query, 
					items)
		DBmanager.con.commit()

