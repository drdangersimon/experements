import pylab as lab
import h_z
import numpy as np
from database_utils import convert_array
import kuiper

db_class = h_z.MCMCResults('.')
# get all params
param_range = {}
for col in ['Real_SFH', 'Real_age', 'Real_Z']:
    param_range[col] = {}
    for table in db_class.tables:
        query_txt = 'Select DISTINCT %s FROM %s'%(col, table)
        param_range[col][table] = db_class.conn.execute(query_txt).fetchall()
    param_range[col] = np.unique(np.ravel(np.vstack(param_range[col].values())))
# get chains and log_probs
mean, median, max_lik, points = [], [], [], []
for age in param_range['Real_age']:
    for metal in  param_range['Real_Z']:
        chains, prob = [], []
        for table in db_class.tables:
            query_txt = 'Select chains FROM %s Where Real_SFH=? AND Real_age=? AND Real_Z=?'%(table)
            temp_array = db_class.conn.execute(query_txt, (0,age,metal,)).fetchall()
            query_txt = 'Select log_post FROM %s Where Real_SFH=? AND Real_age=? AND Real_Z=?'%(table)
            temp_prob = db_class.conn.execute(query_txt, (0,age,metal,)).fetchall()
            if len(temp_array) > 0:
                chains.append(convert_array(temp_array[0][0]))
                prob.append(convert_array(temp_prob[0][0]))

        if len(chains) > 0:
            chains = np.vstack(chains)[:,:3]
            prob  = np.concatenate(prob)
            mean.append(chains[:,1].mean())
            median.append(np.percentile(chains[:,1], 50))
            max_lik.append(chains[prob.argmax(),1])
            points.append([age, metal])

mean = np.asarray(mean)
median = np.asarray(median)
max_lik = np.asarray(max_lik)
points = np.asarray(points)
#plot hist of mean, median, max_like
lab.figure()
lab.hist(points[:,0], 30)
lab.title('Real distribution')
lab.xlable('log(Age)')
#mean
lab.figure()
lab.hist(mean[:,0], 30)
lab.title('Mean distribution')
lab.xlable('log(Age)')
# median
lab.figure()
lab.hist(median[:,0], 30)
lab.title('Median distribution')
lab.xlable('log(Age)')
# max lik
lab.figure()
lab.hist(max_lik[:,0], 30)
lab.title('Maximum likelihood distribution')
lab.xlable('log(Age)')
#plot dt of those
# split randomly into pairs and take difference. make positive
lab.figure()
lab.hist(np.abs(np.diff(np.random.permutation(points[:,0]))), 30)
lab.title('Real $\delta$t')
lab.figure()
lab.hist(np.abs(np.diff(np.random.permutation(mean))), 30)
lab.title('Mean $\delta$t')
lab.figure()
lab.hist(np.abs(np.diff(np.random.permutation(median))), 30)
lab.title('Median $\delta$t')
lab.figure()
lab.hist(np.abs(np.diff(np.random.permutation(max_lik))), 30)
lab.title('Max lik $\delta$t')
# quantify see if simmilar to actual
kuiper.kuiper_two
for test in  [mean, median, max_lik]:
    pass
