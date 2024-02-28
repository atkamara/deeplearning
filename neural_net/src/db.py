from .utils import *

from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer,Float, String, DateTime
from sqlalchemy import create_engine,ForeignKey
from sqlalchemy.orm import sessionmaker

###################################
#                                 #
# Here goes code for sql orm      #
#                                 #
#                                 # 
###################################


Base = declarative_base()

def add_name_id(cl):
	setattr(cl,cl.__name__+'_id',Column(Integer))
	return cl

class DefaultTable:
	id = Column(Integer, primary_key=True, autoincrement=True)
	created_at = Column(DateTime,default=datetime.datetime.utcnow)
	updated_at = Column(DateTime,default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
	name = Column(String)
	


class Architecture(DefaultTable,Base):
	__tablename__ = 'Architecture'

class Layer(DefaultTable,Base):
	__tablename__ = 'Layer'
	Architecture_id = Column(Integer,ForeignKey("Architecture.Architecture_id")) 
	n_in = Column(Integer)
	n_out = Column(Integer)
	#init_method = Column(String)


class Neurons(DefaultTable,Base):
	__tablename__ = 'Neurons'
	Layer_id = Column(Integer,ForeignKey("Layer.Layer_id")) 


class Cost(DefaultTable,Base):
	__tablename__ = 'Cost'
	Architecture_id = Column(Integer,ForeignKey("Architecture.Architecture_id"))
	value = Column(Integer)

class Weight(DefaultTable,Base):
	__tablename__ = 'Weight'
	value = Column(Integer)
	Neurons_id = Column(Integer,ForeignKey("Neurons.Neurons_id")) 


tables = {cl.__name__:[
							(m:=add_name_id(cl)), 
							[k.key for k in m.__table__.columns if k.key not in ['id','created_at','updated_at']
					
					]] for cl in [Architecture,Layer,Neurons,Cost,Weight]
					}

def get_instance(self) -> DefaultTable:
	"""
    Returns an SQLAlchemy Table object corresponding to the given table name.

    Returns:
        Table: SQLAlchemy Table object corresponding to the specified table name.
	"""
	table,cols = tables[str(self)]
	values = {k:v for k,v in self.id.items() if k in cols}
	return table(**values)

def update_instance(self) -> None:
	"""
	Updates the given instance with the provided keyword arguments.

	"""
	_,cols = tables[str(self)]
	for k,v in self.id.items():
		if k in cols:
			setattr(self.table,k,v)


class DBmanager:
	"""
    Manages database connections and sessions using SQLAlchemy.

    Args:
        db (str, optional): Path to database server or SQLite database file. Defaults to None.

    Attributes:
        session (Session): SQLAlchemy session for database operations.

    Methods:
		_DBmanager__start()-> None:
			Starts a session
        add_table(table: DefaultTable) -> None:
            Adds a table instance to the current session.
        commit() -> None:
            Commits changes to the session.
    """
	engines = {}
	status = False

	def __start(db : str=None)-> None:
		db_path = db or f'sqlite:///{get_module_path(["run",f"model{now()}.db"])}'
		DBmanager.path =db_path
		DBmanager.engines[DBmanager.path] = create_engine(DBmanager.path)
		Base.metadata.create_all(DBmanager.engines[DBmanager.path])
		Session = sessionmaker(bind=DBmanager.engines[DBmanager.path])
		DBmanager.session = Session()
			
	def add_table(self,table:DefaultTable) -> None:
		if not DBmanager.status : 
			DBmanager._DBmanager__start()
			DBmanager.status = True
		DBmanager.session.add(table)

	def commit(self) -> None:
		DBmanager.session.commit()