import cPickle as pik
import numpy as np
import pylab as lab
import nedwright
import io, re
import os
import sqlite3 as sql
import glob
import emcee
from database_utils import numpy_sql, adapt_array, convert_array

class BaseData(object):
    '''Telescope like object that takes parameters and returns recoverd parameters'''
    def __init__(self, path):
        '''loads in all .pik files'''
        #maybe make a db?
        raise NotImplementedError

    def __call__(self, params):
        '''Takes in paramters and returns postieriors, map, mle or what ever
        for output'''
        raise NotImplementedError
    
class GetHZ(object):
    '''Generates H(Z) with noise observables'''
    def __init__(self,):
        pass
    
class RecoverHZ(object):
    '''Uses 3 different ways of recovering the H(z).
    1. t(z)=H(z) * something
    2. H(z) = dt/dz
    3. same as 2 but stack galaxies'''

class MCMCResults(BaseData):
    '''loads data from emcee run. Files may be long so will put into db'''
    def __init__(self, path, reference_list_path='reference_list.txt',
                 out_db='mc_results.db'):
        '''loads data and makes db'''
        # get all results*.pik files
        files = glob.glob(os.path.join(path, 'results*.pik'))
        # make db for quick access
        if not os.path.exists(out_db):
            reference = np.loadtxt(reference_list_path)
            # create new dbs
            self.db = numpy_sql(out_db)
            self.conn = self.db.cursor()
            for f in files:
                # get result number
                num = ''.join(re.findall('\d',f))
                self.conn.execute('''CREATE TABLE s%s (Real_SFH real, Real_age real, Real_Z real, chains array, log_post array)'''%num)
                results = pik.load(open(f))
                # put in db
                for res in results:
                    if len(results[res]) < 1:
                        continue
                    samp = results[res][0]
                    row = (results[res][2][0], results[res][2][1], results[res][2][2],
                        adapt_array(samp.flatchain),adapt_array(samp.flatlnprobability))
                    self.conn.execute('INSERT INTO s%s VALUES (?,?,?,?,?)'%num, row)
                
                self.conn.execute('CREATE UNIQUE INDEX i%s ON s%s (Real_SFH, Real_age, Real_Z)'%(num, num))
                self.db.commit()
        else:
            self.db = numpy_sql(out_db)
            self.conn = self.db.cursor()
        # get tables
        self.tables = []
        for i in self.conn.execute('select * from sqlite_master').fetchall():
            if i[0] == 'table':
                self.tables.append(i[1])
        

    def __call__(self, param):
        '''put paratemter and will return postierors'''
        # find nearest values
        self.conn.execute('Select * From * Where Real_SFH  Real_age, Real_Z')

    def _grid_search(self, points)
        '''Finds points that make a cube around input point and returns them with
        their spectra'''
        if not hasattr(self, 'param_range'):
            query_txt = 'Setect 
            self.param_range = 
        
