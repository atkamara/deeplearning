"""
Provides utility functions

- `get_module_path` - Returns the path to a subdirectory named 'dir' relative to the currently executed script.
- `now` - current timestamp
- `unfold` - Unfolds a nested dictionary by appending the values of inner dictionaries to the outer dictionary.
"""
import numpy,os,pandas,re
import datetime
from tqdm.autonotebook import tqdm
from typing import Literal
import requests
from typing import Union
import matplotlib.pyplot as plt

class Boostrap:
    def __init__(self,data:Union[numpy.ndarray,tuple],sample_size:int=100,n_sample:int=100):
        self.not_tuple = False
        if isinstance(data,numpy.ndarray): 
            self.not_tuple = True
            data = (data)
        self.data = data
        self.n = len(self.data[0])
        self.sample_size = sample_size
        self.c = 0
        self.n_sample = n_sample
        self.ix = numpy.random.randint(0,self.n,size=(self.n_sample,self.sample_size))
    def __len__(self):
        return self.n_sample
    def __iter__(self) : 
        return self
    def __next__(self):
        if self.c < self.n_sample:
            self.c+=1
            ix = self.ix[self.c-1]
            out = tuple(d[ix,:] for d in self.data)
            if self.not_tuple:
                out = out[0]
            return out
        self.c = 0
        raise StopIteration


class Pearson:
    """
    Computes the Pearson correlation matrix and generates a heatmap.

    Attributes:
        X : numpy.ndarray
            The input data containing features.
        n : int
            number of observations
        k: int
            number of features
        cols: list
            list of fed columns
        cov: numpy.ndarray
            covariance matrix
        var: numpy.ndarray
            variance matrix
        corr: numpy.ndarray
            correlation matrix
        
    Methods:
        __init__(X:numpy.ndarray,cols:list=None)->None:
            Initialize Pearson object
        corr()->numpy.ndarray:
            computes Pearson correlation matrix
        heatmap(ax=None,fontsize:int=6,digits:int=1, xrotation:Union[int,str]=45,yrotation:Union[int,str]='horizontal') -> None:
            plots correlation heatmap

    Example:
        ```python
            >>> import pandas as pd
            >>> from matplotlib import pyplot as plt
            >>>
            >>> # Create a sample dataset
            >>> data = pd.DataFrame({
            ...     'feature1': [1, 2, 3, 4, 5],
            ...     'feature2': [5, 4, 3, 2, 1],
            ...     'feature3': [3, 3, 3, 3, 3]
            ... })
            >>>
            >>> # Initialize the Pearson correlation analyzer
            >>> pearson_analyzer = Pearson(X=data.values,cols=data.columns)
            >>> # Compute the Pearson correlation matrix
            >>> correlation_matrix = pearson_analyzer.corr()
            >>> # Generate the heatmap
            >>> pearson_analyzer.heatmap()
        ```
    """
    def __init__(self,X:numpy.ndarray,cols:list=None) -> None:
        """
        Initialize Pearson object

        Args:
            X:numpy.ndarray
                array for which you'd like to get correlations from
            cols: list
                list of labels for columns
        
        """
        self.X = X
        self.n,self.k = X.shape
        self.cols = cols or tuple(range(self.k))
    def corr(self) -> numpy.ndarray:
        self.cov = (v:=(self.X - self.X.mean(axis=0))).T.dot(v)/self.n
        self.var = (std:=self.X.std(axis=0)).reshape(-1,1)*std
        self.corr = self.cov/self.var
        return self.corr
    def heatmap(self,ax=None,fontsize:int=6,digits:int=1, xrotation:Union[int,str]=45,yrotation:Union[int,str]='horizontal'):    
        ax = ax or  plt.subplots()[1]
        im = ax.imshow(self.corr)
        im.set_clim(-1, 1)
        ax.grid(False)
        ax.xaxis.set_ticks(ticks=tuple(range(self.k)), labels=self.cols,rotation=xrotation)
        ax.yaxis.set_ticks(ticks=tuple(range(self.k)), labels=self.cols,rotation=yrotation)
        for i in range(self.k):
            for j in range(self.k):
                ax.text(j, i, self.corr.round(digits)[i, j], ha='center', va='center',fontsize=fontsize,color='r')
        cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
        plt.show()
def make_circle_data(centers:list,radii:list,p:float=.2,n_grid:int=100,xmin:int=-100,xmax:int=100,ymin:int=-100,ymax:int=100)->tuple:
    """
    Generates random data points distributed within circles.

    Parameters:
        centers : list of tuples
            List of (x, y) coordinates representing the centers of circles.
        radii : list of floats
            List of radii for each circle.
        p : float, optional (default=0.2)
            Percentage of randomly picked data points outside the circles.
        n_grid : int, optional (default=100)
            Meshgrid parameter for creating the grid of points.
        xmin : int, optional (default=-100)
            Minimum x limit for the data points.
        xmax : int, optional (default=100)
            Maximum x limit for the data points.
        ymin : int, optional (default=-100)
            Minimum y limit for the data points.
        ymax : int, optional (default=100)
            Maximum y limit for the data points.

    Returns:
        X : numpy.ndarray, shape (n_samples, 2)
            2D matrix of features (coordinates of data points).
        y : numpy.ndarray, shape (n_samples,1)
            Labels corresponding to the data points (1 if inside a circle, 0 otherwise).

    Example:
        ```python
            >>> centers = [(0, 0), (20, 30)]
            >>> radii = [10, 15]
            >>> X, y = make_circle(centers, radii, p=0.1)
            >>> print(X.shape, y.shape)
            (200, 2) (200,1)
        ```
    """
    x,y = numpy.linspace(xmin,xmax,n_grid),numpy.linspace(ymin,ymax,n_grid)
    xm,ym = numpy.meshgrid(x,y)
    x_news,y_news = [],[]
    labels = []
    n_centers = len(centers)
    j = 0
    for i in range(n_centers):
        c = (xm-centers[i][0])**2 + (ym-centers[i][1])**2 <= radii[i]**2
        x_new,y_new = numpy.where(c)
        n = len(x_new)
        n_sample = int(numpy.ceil(n*p))
        ix = numpy.random.randint(0,n,n_sample)
        x_news += [x[x_new][ix]]
        y_news += [y[y_new][ix]]
        labels += [j]*n_sample
        j+=1
    x_news = numpy.concatenate(x_news)
    y_news = numpy.concatenate(y_news)
    labels = numpy.array(labels).reshape(-1,1)
    return numpy.c_[x_news,y_news],labels

class IrisDatasetDownloader:
    """
        Downloads the Iris dataset from an online CSV source.

    
        Parameters:
            src : str
                URL to the online CSV file containing the Iris dataset default at https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv
    
        Attributes:
            target_names : list of str
                Names of each target label (species) in the dataset.
            feature_names : list of str
                Names of all features (attributes) in the dataset.
            csv : str
                raw CSV textfile
            description : str
                Full description of iris database
            data : numpy.ndarray
                array of all features
            target : numpy.ndarray
                array of target variable
    
        Example:
            ```python
                >>> iris = IrisDatasetDownloader()
                >>> iris.load_dataset()
                >>> print(iris.data.shape,iris.target.shape)
                (150, 4) (150, 1)
                >>> print(iris.target_names)
                ['setosa', 'versicolor', 'virginica']
                >>> print(iris.feature_names)
                ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
                >>> print(iris.data[:5,:])
                [[5.1 3.5 1.4 0.2]
                [4.9 3.  1.4 0.2]
                [4.7 3.2 1.3 0.2]
                [4.6 3.1 1.5 0.2]
                [5.  3.6 1.4 0.2]]
                >>> print(iris.target[:5,:])
                [[0]
                 [0]
                 [0]
                 [0]
                 [0]]
            ```
    """
    description = r"""
        1. Title: Iris Plants Database
            Updated Sept 21 by C.Blake - Added discrepency information

        2. Sources:
            (a) Creator: R.A. Fisher
            (b) Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
            (c) Date: July, 1988

        3. Past Usage:
            - Publications: too many to mention!!!  Here are a few.
        1. Fisher,R.A. "The use of multiple measurements in taxonomic problems"
            Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions
            to Mathematical Statistics" (John Wiley, NY, 1950).
        2. Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
            (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
        3. Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
            Structure and Classification Rule for Recognition in Partially Exposed
            Environments".  IEEE Transactions on Pattern Analysis and Machine
            Intelligence, Vol. PAMI-2, No. 1, 67-71.
            -- Results:
                -- very low misclassification rates (0% for the setosa class)
        4. Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE 
            Transactions on Information Theory, May 1972, 431-433.
            -- Results:
                -- very low misclassification rates again
        5. See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al's AUTOCLASS II
            conceptual clustering system finds 3 classes in the data.

        4. Relevant Information:
            --- This is perhaps the best known database to be found in the pattern
                recognition literature.  Fisher's paper is a classic in the field
                and is referenced frequently to this day.  (See Duda & Hart, for
                example.)  The data set contains 3 classes of 50 instances each,
                where each class refers to a type of iris plant.  One class is
                linearly separable from the other 2; the latter are NOT linearly
                separable from each other.
            --- Predicted attribute: class of iris plant.
            --- This is an exceedingly simple domain.
            --- This data differs from the data presented in Fishers article
                (identified by Steve Chadwick,  spchadwick@espeedaz.net )
                The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa"
                where the error is in the fourth feature.
                The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa"
                where the errors are in the second and third features.  

        5. Number of Instances: 150 (50 in each of three classes)

        6. Number of Attributes: 4 numeric, predictive attributes and the class

        7. Attribute Information:
            1. sepal length in cm
            2. sepal width in cm
            3. petal length in cm
            4. petal width in cm
            5. class: 
                -- Iris Setosa
                -- Iris Versicolour
                -- Iris Virginica

        8. Missing Attribute Values: None

        Summary Statistics:
                    Min  Max   Mean    SD   Class Correlation
           sepal length: 4.3  7.9   5.84  0.83    0.7826   
            sepal width: 2.0  4.4   3.05  0.43   -0.4194
           petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
            petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)

        9. Class Distribution: 33.3% for each of 3 classes.
        """
    def __init__(self,src="https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"):
        self.src = src
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.target_names = []
        self.csv = None
    def load_dataset(self):
        """
        Load the Iris dataset from the specified online CSV source.

        """
        self.csv = requests.get(self.src).text
        columns,*data,_ = self.csv.split('\n')
        data,target = [r.split(',')[:-1] for r in data],[r.split(',')[-1] for r in data ]
        self.target_names = []
        for l in target:
            if l not in self.target_names:
                self.target_names += [l]
        self.target = numpy.array([[self.target_names.index(l)] for l in target])
        self.feature_names = columns.split(',')[:-1]
        for i in range(len(self.feature_names)):
            self.feature_names[i] = self.feature_names[i].replace('_',' ')
            self.feature_names[i] +=' (cm)'
        self.data = numpy.array(data).astype(float)

def get_module_path(dir: list[str]) -> str:
    """
    Returns the path to a subdirectory named 'dir' relative to the currently executed script.

    Args:
        dir (str): path to the subdirectory.

    Returns:
        str: Absolute path to the specified subdirectory.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),*dir)

def now()-> int:
    """
    Returns the current timestamp as an integer.

    Returns:
        int: Current timestamp (number of seconds since the epoch).
    """
    return int(datetime.datetime.now().timestamp())

def unfold(d: dict) -> dict:
    """
    Unfolds a nested dictionary by appending the values of inner dictionaries to the outer dictionary.

    Args:
        d (dict): Input dictionary with nested dictionaries.

    Returns:
        dict: Unfolded dictionary with concatenated keys.
    
    Example:
    ```python
            >>> d = {'a':1,'b':{'c':2,'d':4}}
            >>> unfold(d)
            {'a': 1, 'b_c': 2, 'b_d': 4}
    ```
    """
    new_d = {}
    for k in d:
        if hasattr(d[k],'keys'):
            for j in d[k]:
                new_d[f'{k}_{j}'] = d[k][j]
        else:
            new_d[k] = d[k]
    return new_d
class HouseDatasetDownloader:
    """
        Downloads boston housing dataset.

    
        Parameters:
            src : str
                URL to the online CSV file containing the Iris dataset default at https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv
    
        Attributes:
            columns : list of str
                ordered list of columns in the dataset.
            data : array
                data array of features and target variable
            csv : str
                Raw CSV textfile
            description : str
                Full description of housing database
    
        Example:
            ```python
                >>> houseloader = HouseDatasetDownloader()
                >>> houseloader.load_dataset()
                >>> print(houseloader.columns)
                ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
                >>> print(houseloader.data)
                [[6.3200e-03 1.8000e+01 2.3100e+00 ... 3.9690e+02 4.9800e+00 2.4000e+01]
                [2.7310e-02 0.0000e+00 7.0700e+00 ... 3.9690e+02 9.1400e+00 2.1600e+01]
                [2.7290e-02 0.0000e+00 7.0700e+00 ... 3.9283e+02 4.0300e+00 3.4700e+01]
                ...
                [6.0760e-02 0.0000e+00 1.1930e+01 ... 3.9690e+02 5.6400e+00 2.3900e+01]
                [1.0959e-01 0.0000e+00 1.1930e+01 ... 3.9345e+02 6.4800e+00 2.2000e+01]
                [4.7410e-02 0.0000e+00 1.1930e+01 ... 3.9690e+02 7.8800e+00 1.1900e+01]]
            ```
    """

    def __init__(self,src="http://lib.stat.cmu.edu/datasets/boston"):
        self.src = src
        self.csv = None
    def load_dataset(self):
        """
        Load the House dataset from the specified online CSV source.

        """
        self.csv = requests.get(self.src).text
        self.description,data = self.csv[:self.csv.index("0.00632")],self.csv[self.csv.index("0.00632"):]
        data = data.replace('\n  ','  ').split('\n')[:-1]
        data = [re.findall(r'(\d+\.*\d*)',r) for r in data]
        self.data = numpy.array(data).astype(float)
        self.columns = columns = [c.split()[0] for c in self.description.split('\n')[-16:-2]]
