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
from LRG_lik import grid_search

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

    def _grid_search(self, points):
        '''Finds points that make a cube around input point and returns them with
        their spectra'''
        if not hasattr(self, 'param_range'):
            # get unique points
            param_range = {}
            self.param_range = {}
            for col in ['Real_SFH', 'Real_age', 'Real_Z']:
                param_range[col] = {}
                for table in self.tables:
                    query_txt = 'Select DISTINCT %s FROM %s'%(col, table)
                    param_range[col][table] = self.conn.execute(query_txt).fetchall()
                self.param_range[col] = np.unique(np.ravel(np.vstack(param_range[col].values())))

        # find all talbes with values
        interp_points = grid_search(points, self.param_range.values())
        # combine all chains with interp values
        chains = {}
        pdf = {}
        for index, p in enumerate(interp_points):
            chains[index] = []
            for table in self.tables:
                query_txt = 'Select chains FROM %s Where Real_SFH=? AND Real_age=? AND Real_Z=?'%(table)
                temp_array = self.conn.execute(query_txt, p).fetchall()
                if len(temp_array) > 0:
                    chains[index].append(convert_array(temp_array[0][0]))

            chains[index] = np.vstack(chains[index])[:,:3]
            # make pdf
            pdf[index] = bayesian_blocks(chains[index][:,0])
        # interpolate pdfs



#### from AstroML                    
from scipy.special import gammaln
from scipy import optimize
def scotts_bin_width(data, return_bins=False):
    r"""Return the optimal histogram bin width using Scott's rule:

    Parameters
    ----------
    data : array-like, ndim=1
        observed (one-dimensional) data
    return_bins : bool (optional)
        if True, then return the bin edges

    Returns
    -------
    width : float
        optimal bin width using Scott's rule
    bins : ndarray
        bin edges: returned if `return_bins` is True

    Notes
    -----
    The optimal bin width is

    .. math::
        \Delta_b = \frac{3.5\sigma}{n^{1/3}}

    where :math:`\sigma` is the standard deviation of the data, and
    :math:`n` is the number of data points.

    See Also
    --------
    knuth_bin_width
    freedman_bin_width
    astroML.plotting.hist
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    sigma = np.std(data)

    dx = 3.5 * sigma * 1. / (n ** (1. / 3))

    if return_bins:
        Nbins = np.ceil((data.max() - data.min()) * 1. / dx)
        Nbins = max(1, Nbins)
        bins = data.min() + dx * np.arange(Nbins + 1)
        return dx, bins
    else:
        return dx


def freedman_bin_width(data, return_bins=False):
    r"""Return the optimal histogram bin width using the Freedman-Diaconis rule

    Parameters
    ----------
    data : array-like, ndim=1
        observed (one-dimensional) data
    return_bins : bool (optional)
        if True, then return the bin edges

    Returns
    -------
    width : float
        optimal bin width using Scott's rule
    bins : ndarray
        bin edges: returned if `return_bins` is True

    Notes
    -----
    The optimal bin width is

    .. math::
        \Delta_b = \frac{2(q_{75} - q_{25})}{n^{1/3}}

    where :math:`q_{N}` is the :math:`N` percent quartile of the data, and
    :math:`n` is the number of data points.

    See Also
    --------
    knuth_bin_width
    scotts_bin_width
    astroML.plotting.hist
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    if n < 4:
        raise ValueError("data should have more than three entries")

    dsorted = np.sort(data)
    v25 = dsorted[n / 4 - 1]
    v75 = dsorted[(3 * n) / 4 - 1]

    dx = 2 * (v75 - v25) * 1. / (n ** (1. / 3))

    if return_bins:
        Nbins = np.ceil((dsorted[-1] - dsorted[0]) * 1. / dx)
        Nbins = max(1, Nbins)
        bins = dsorted[0] + dx * np.arange(Nbins + 1)
        return dx, bins
    else:
        return dx


class KnuthF(object):
    r"""Class which implements the function minimized by knuth_bin_width

    Parameters
    ----------
    data : array-like, one dimension
        data to be histogrammed

    Notes
    -----
    the function F is given by

    .. math::
        F(M|x,I) = n\log(M) + \log\Gamma(\frac{M}{2})
        - M\log\Gamma(\frac{1}{2})
        - \log\Gamma(\frac{2n+M}{2})
        + \sum_{k=1}^M \log\Gamma(n_k + \frac{1}{2})

    where :math:`\Gamma` is the Gamma function, :math:`n` is the number of
    data points, :math:`n_k` is the number of measurements in bin :math:`k`.

    See Also
    --------
    knuth_bin_width
    astroML.plotting.hist
    """
    def __init__(self, data):
        self.data = np.array(data, copy=True)
        if self.data.ndim != 1:
            raise ValueError("data should be 1-dimensional")
        self.data.sort()
        self.n = self.data.size

    def bins(self, M):
        """Return the bin edges given a width dx"""
        return np.linspace(self.data[0], self.data[-1], int(M) + 1)

    def __call__(self, M):
        return self.eval(M)

    def eval(self, M):
        """Evaluate the Knuth function

        Parameters
        ----------
        dx : float
            Width of bins

        Returns
        -------
        F : float
            evaluation of the negative Knuth likelihood function:
            smaller values indicate a better fit.
        """
        M = int(M)

        if M <= 0:
            return np.inf

        bins = self.bins(M)
        nk, bins = np.histogram(self.data, bins)

        return -(self.n * np.log(M)
                 + gammaln(0.5 * M)
                 - M * gammaln(0.5)
                 - gammaln(self.n + 0.5 * M)
                 + np.sum(gammaln(nk + 0.5)))


def knuth_bin_width(data, return_bins=False):
    r"""Return the optimal histogram bin width using Knuth's rule [1]_

    Parameters
    ----------
    data : array-like, ndim=1
        observed (one-dimensional) data
    return_bins : bool (optional)
        if True, then return the bin edges

    Returns
    -------
    dx : float
        optimal bin width. Bins are measured starting at the first data point.
    bins : ndarray
        bin edges: returned if `return_bins` is True

    Notes
    -----
    The optimal number of bins is the value M which maximizes the function

    .. math::
        F(M|x,I) = n\log(M) + \log\Gamma(\frac{M}{2})
        - M\log\Gamma(\frac{1}{2})
        - \log\Gamma(\frac{2n+M}{2})
        + \sum_{k=1}^M \log\Gamma(n_k + \frac{1}{2})

    where :math:`\Gamma` is the Gamma function, :math:`n` is the number of
    data points, :math:`n_k` is the number of measurements in bin :math:`k`.

    References
    ----------
    .. [1] Knuth, K.H. "Optimal Data-Based Binning for Histograms".
           arXiv:0605197, 2006

    See Also
    --------
    KnuthF
    freedman_bin_width
    scotts_bin_width
    """
    knuthF = KnuthF(data)
    dx0, bins0 = freedman_bin_width(data, True)
    M0 = len(bins0) - 1
    M = optimize.fmin(knuthF, len(bins0))[0]
    bins = knuthF.bins(M)
    dx = bins[1] - bins[0]

    if return_bins:
        return dx, bins
    else:
        return dx


def histogram(a, bins=10, range=None, **kwargs):
    """Enhanced histogram

    This is a histogram function that enables the use of more sophisticated
    algorithms for determining bins.  Aside from the `bins` argument allowing
    a string specified how bins are computed, the parameters are the same
    as numpy.histogram().

    Parameters
    ----------
    a : array_like
        array of data to be histogrammed

    bins : int or list or str (optional)
        If bins is a string, then it must be one of:
        'blocks' : use bayesian blocks for dynamic bin widths
        'knuth' : use Knuth's rule to determine bins
        'scotts' : use Scott's rule to determine bins
        'freedman' : use the Freedman-diaconis rule to determine bins

    range : tuple or None (optional)
        the minimum and maximum range for the histogram.  If not specified,
        it will be (x.min(), x.max())

    other keyword arguments are described in numpy.hist().

    Returns
    -------
    hist : array
        The values of the histogram. See `normed` and `weights` for a
        description of the possible semantics.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.

    See Also
    --------
    numpy.histogram
    astroML.plotting.hist
    """
    a = np.asarray(a)

    # if range is specified, we need to truncate the data for
    # the bin-finding routines
    if (range is not None and (bins in ['blocks', 'knuth',
                                        'scotts', 'freedman'])):
        a = a[(a >= range[0]) & (a <= range[1])]

    if bins == 'blocks':
        bins = bayesian_blocks(a)
    elif bins == 'knuth':
        da, bins = knuth_bin_width(a, True)
    elif bins == 'scotts':
        da, bins = scotts_bin_width(a, True)
    elif bins == 'freedman':
        da, bins = freedman_bin_width(a, True)
    elif isinstance(bins, str):
        raise ValueError("unrecognized bin code: '%s'" % bins)

    return np.histogram(a, bins, range, **kwargs)
