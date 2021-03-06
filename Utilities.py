import numpy as np
import glob
import csv
import os
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd  
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from scipy.signal import gaussian
from scipy.ndimage import filters
import csv
import re
from scipy.ndimage import gaussian_filter1d
from sklearn import metrics

#### Load & modify data #####

def load_filelist(folder_path):
    """
    Loads list of file paths given a folder path.
    params:
        folder_path (string): path to file.
    returns:
        flist (list): list of file paths.  
    """
    flist = glob.glob(folder_path)
    flist.remove("/home/april/IActData_Export/R3MK120709_02.txt")
    return flist

def load_table(filepath, rm_cap, scale, zero):
    """
    Loads table written as tsv file into a list of numbers and voltages for a single cell.
    params:
        filepath (string): path to file.
    returns:
        table (array): Table of data for single cell.
        voltage_list (array): Voltage list for a single cell. 
    """
    table = []
    with open(filepath) as tsv:
        for i, line in enumerate(csv.reader(tsv, delimiter="\t")):
            if i == 0:
                voltage_list = line
            else:
                line = [np.float(item) for item in line]
                table.append(line)
    table = np.array(table)
    if rm_cap:
        table = np.delete(table,slice(6000,6100),0)
        table = np.delete(table,slice(1000,1100),0)
    voltage_list = np.array(voltage_list)
    if scale:
        ds = 500
        de = 500
        len_uptime = 10800 -ds -de
        meantimes = table[ds+len_uptime//3:-de,:]
        uptimes =table[ds:-de,:]
        mean_uptimes = np.mean(meantimes, axis=0)
        table[ds:-de,:] = uptimes / mean_uptimes
    if zero:
        start = 750
        end = 800
        zerotimes = table[start:end,:]
        mean_zerotimes = np.mean(zerotimes, axis=0)
        table = table - mean_zerotimes
    return(table, voltage_list)
    

def generate_table(flist, rm_cap, norm=5):
    """
    Loads table from a given file list
    params:
        flist (list): List of file paths for individual cells.
        rm_cap: Flag to remove capacitance artefact
        return_flist: Flag to return file list or not.
    returns:
        cell_table (np.array): Array of cells data. 
    """
    tables = []
    voltage_lists = []
    final_flist = []
    for path in flist:
        my_table, my_voltage_list = load_table(path, rm_cap=rm_cap, scale=False, zero=True)
        if rm_cap & (my_table.shape == (10800, 16)):
            tables.append(my_table)
            voltage_lists.append(my_voltage_list)
            final_flist.append(re.sub('.txt','',os.path.basename(path)))
        if ~rm_cap & (my_table.shape == (11000, 16)):
            tables.append(my_table)
            voltage_lists.append(my_voltage_list)
            final_flist.append(re.sub('.txt','',os.path.basename(path)))
    table = np.stack(tables)
    if rm_cap:
        table = table.reshape((table.shape[0],(10800*16)), order="F")
    else: 
        table = table.reshape((table.shape[0],(11000*16)), order="F")
    if norm == 0: 
        normed_table = np.zeros_like(table)
        for i, trace in enumerate(table):
            trace = (trace - np.min(trace))
            normed_table[i]  = trace / np.mean(trace[0:50])
            normed_table[i] = normed_table[i] - 1
        return(final_flist, normed_table) 
    if norm == 1: 
        normed_table = np.zeros_like(table)
        for i, trace in enumerate(table):
            trace = (trace - np.min(trace))
            normed_table[i]  = trace / np.mean(trace[163000:167000])
        return(final_flist, normed_table)
    if norm == 2:
        normed_table = np.zeros_like(table)
        cells_max = np.amax(table, axis=1)
        normed_table = table / cells_max[:,None]
        return(final_flist, normed_table)
    else:
        return(final_flist, table)
        
def generate_table_nonorm(flist, rm_cap=True):
    """
    Loads table from a given file list
    params:
        flist (list): List of file paths for individual cells.
        rm_cap: Flag to remove capacitance artefact
        return_flist: Flag to return file list or not.
    returns:
        cell_table (np.array): Array of cells data. 
    """
    tables = []
    voltage_lists = []
    final_flist = []
    for path in flist:
        my_table, my_voltage_list = load_table(path, rm_cap=rm_cap, scale=False, zero=True)
        if rm_cap & (my_table.shape == (10800, 16)):
            tables.append(my_table)
            voltage_lists.append(my_voltage_list)
            final_flist.append(re.sub('.txt','',os.path.basename(path)))
        if ~rm_cap & (my_table.shape == (11000, 16)):
            tables.append(my_table)
            voltage_lists.append(my_voltage_list)
            final_flist.append(re.sub('.txt','',os.path.basename(path)))
    table = np.stack(tables)
    if rm_cap:
        table = table.reshape((table.shape[0],(10800*16)), order="F")
    else: 
        table = table.reshape((table.shape[0],(11000*16)), order="F")
    return(final_flist, table)
        
def readin_cell_labels(path='data/Cell_Types.csv'):
    '''
    Reads in metadata for each cell including name, manual cell classification, and whether it is excluded from the dataset
    Parameters:
        Path (str): Path to .csv file containing metadata
    Returns:
        cell_labels (2d array): n_cell by 3 array IDing each cel by name, classification, and boolean indicating it's been removed from the dataset
    
    '''
    cell_labels = []
    with open(path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            #cell_labels.append(np.array([row['Cell names'], row['Type'], row['Cut?']])) #past formatting
            cell_labels.append(row)
    cell_labels = np.array(cell_labels)
    return(cell_labels)
    
def filter_goodlabels(cell_labels, table):
    good_labels = ['DB4','DB5','DB6','IMB','BB','RB','DB1','FMB','DB2','DB3']
    keep_flag = np.ones(len(cell_labels[:,1]), dtype=bool)
    for i, cell in enumerate(cell_labels[:,1]):
        if cell in good_labels:
            keep_flag[i] = True
        else:
            keep_flag[i] = False
    cell_labels = cell_labels[keep_flag]
    table = table[keep_flag]
    return(cell_labels, table, keep_flag)
    
def check_flist_match(final_flist, cell_labels):
    '''
    Make sure the filenames we loaded in are the same and in the same order as our ground truth data
    '''
    if(np.alltrue(np.array(final_flist)==cell_labels[:,0])):
        print(f'All {len(final_flist)} traces match!')
    else:
        print(f'Warning! filelist does not match loaded in files! Neuron Assignments will be incorrect!!!')
    
def mark_cell_categories(cell_labels, categorization):
    '''
    Create Cell Categorizations that are interesting biologically.
    Parameters:
        cell_labels (2d array) numpy array where [:,1] is an array of cell type labels
        categorization (string) name of the type of catetgorization we'd like to do. Only specific ones supported
    Returns:
        cat (1d array of int) numpy array of len(cell_labels) putting each cell into a category
        cat_labels (1d array of strings) numpy array of len(unique(cell_labels))) naming each category
    '''
    cat = np.array([-1 for label in cell_labels])
    if(categorization=='On_Off'):
        #strings defining categories
        off_list = ['DB1','FMB','DB2','DB3','OFF']
        on_list = ['DB4','DB5','DB6','IMB','BB','RB','ON']
        other_list = ['Unknown', 'DB','AC','MB','GBC']
        cat_labels = ['On','Off','Other']
        #find indexes
        on_idx = np.where([any(on in cell for on in on_list) for cell in cell_labels[:,1]])
        off_idx = np.where([any(off in cell for off in off_list) for cell in cell_labels[:,1]])
        other_idx = np.where([any(oth == cell for oth in other_list) for cell in cell_labels[:,1]])
        #apply categories to list
        cat[on_idx] = 0
        cat[off_idx] = 1
        cat[other_idx] = 2
        
    elif(categorization=='DB'):
        #strings defining categories
        cat_labels = ['DB','Non-DB']
        #find indexes
        db_idx = np.where(['DB' in cell for cell in cell_labels[:,1]])
        ndb_idx = np.where(np.invert(['DB' in cell for cell in cell_labels[:,1]]))
        #apply categories to list
        cat[db_idx] = 0
        cat[ndb_idx] = 1
        
    elif(categorization=='RBC'):
        #strings defining categories
        cat_labels = ['RBC','Non-RBC']
        #find indexes
        rb_idx = np.where(['RBC' in cell for cell in cell_labels[:,1]])
        nrb_idx = np.where(np.invert(['RBC' in cell for cell in cell_labels[:,1]]))
        #apply categories to list
        cat[rb_idx] = 0
        cat[nrb_idx] = 1
        
    elif(categorization=='DB_Subtypes'):
        #strings defining categories
       #'AC', 'BB', 'DB', 'DB1', 'DB1 cone', 'DB1 or DB4B', 'DB2',
       #'DB2 cone', 'DB2 or DB3', 'DB2 or FMB', 'DB2 or ON', 'DB3',
       #'DB3 or DB4', 'DB3A', 'DB3B', 'DB3B cone', 'DB4', 'DB4 or DB3B',
       #'DB4 or DB5', 'DB5', 'DB5 or DB6', 'DB6', 'FMB', 'FMB cone',
       #'FMB cones', 'FMB or DB2', 'GBC', 'IMB', 'MB', 'OFF DB', 'ON',
       #'ON DB', 'RBC', 'RBC or DB6', 'RBC or ON', 'Unknown']
        cat_labels = ['Other','DB_unspecified','DB1','DB2','DB3','DB4','DB5','DB6']
        #find indexes
        DBx_idx = np.where(['DB' in cell for cell in cell_labels[:,1]])
        DB1_idx = np.where(['DB1' in cell for cell in cell_labels[:,1]])
        DB2_idx = np.where(['DB2' in cell for cell in cell_labels[:,1]])
        DB3_idx = np.where(['DB3' in cell for cell in cell_labels[:,1]])
        DB4_idx = np.where(['DB4' in cell for cell in cell_labels[:,1]])
        DB5_idx = np.where(['DB5' in cell for cell in cell_labels[:,1]])
        DB6_idx = np.where(['DB6' in cell for cell in cell_labels[:,1]])
        #apply categories to list
        cat[:] = 0 #other
        cat[DBx_idx] = 1
        cat[DB1_idx] = 2
        cat[DB2_idx] = 3
        cat[DB3_idx] = 4
        cat[DB4_idx] = 5
        cat[DB5_idx] = 6
        cat[DB6_idx] = 7
        
    elif(categorization=='Full'):
        #strings defining categories
       #'AC', 'BB', 'DB', 'DB1', 'DB1 cone', 'DB1 or DB4B', 'DB2',
       #'DB2 cone', 'DB2 or DB3', 'DB2 or FMB', 'DB2 or ON', 'DB3',
       #'DB3 or DB4', 'DB3A', 'DB3B', 'DB3B cone', 'DB4', 'DB4 or DB3B',
       #'DB4 or DB5', 'DB5', 'DB5 or DB6', 'DB6', 'FMB', 'FMB cone',
       #'FMB cones', 'FMB or DB2', 'GBC', 'IMB', 'MB', 'OFF DB', 'ON',
       #'ON DB', 'RBC', 'RBC or DB6', 'RBC or ON', 'Unknown']
        cat_labels = ['ON','AC','BB','DB','MB','RBC','GBC','Unknown']
        #find indexes
        on_idx = np.where(['ON' in cell for cell in cell_labels[:,1]])
        ac_idx = np.where(['AC' in cell for cell in cell_labels[:,1]])
        bb_idx = np.where(['BB' in cell for cell in cell_labels[:,1]])
        db_idx = np.where(['DB' in cell for cell in cell_labels[:,1]])
        mb_idx = np.where(['MB' in cell for cell in cell_labels[:,1]])
        rbc_idx = np.where(['RBC' in cell for cell in cell_labels[:,1]])
        gbc_idx = np.where(['GBC' in cell for cell in cell_labels[:,1]])
        unk_idx = np.where(['Unknown' in cell for cell in cell_labels[:,1]])
        #apply categories to list
        cat[on_idx] = 0
        cat[ac_idx] = 1
        cat[bb_idx] = 2
        cat[db_idx] = 3
        cat[mb_idx] = 4
        cat[rbc_idx] = 5
        cat[gbc_idx] = 6
        cat[unk_idx] = 7
        
    elif(categorization=='DB1'):
        #strings defining categories
        cat_labels = ['DB1','Non-DB1']
        #find indexes
        db1_idx = np.where(['DB1' in cell for cell in cell_labels[:,1]])
        ndb1_idx = np.where(np.invert(['DB1' in cell for cell in cell_labels[:,1]]))
        #apply categories to list
        cat[db1_idx] = 0
        cat[ndb1_idx] = 1
        
    elif(categorization=='DB2'):
        #strings defining categories
        cat_labels = ['DB2','Non-DB2']
        #find indexes
        db2_idx = np.where(['DB2' in cell for cell in cell_labels[:,1]])
        ndb2_idx = np.where(np.invert(['DB2' in cell for cell in cell_labels[:,1]]))
        #apply categories to list
        cat[db2_idx] = 0
        cat[ndb2_idx] = 1

    elif(categorization=='DB3'):
        #strings defining categories
        cat_labels = ['DB3','Non-DB3']
        #find indexes
        db3_idx = np.where(['DB3' in cell for cell in cell_labels[:,1]])
        ndb3_idx = np.where(np.invert(['DB3' in cell for cell in cell_labels[:,1]]))
        #apply categories to list
        cat[db3_idx] = 0
        cat[ndb3_idx] = 1        
        
    elif(categorization=='DB4'):
        #strings defining categories
        cat_labels = ['DB4','Non-DB4']
        #find indexes
        db4_idx = np.where(['DB4' in cell for cell in cell_labels[:,1]])
        ndb4_idx = np.where(np.invert(['DB4' in cell for cell in cell_labels[:,1]]))
        #apply categories to list
        cat[db4_idx] = 0
        cat[ndb4_idx] = 1        
        
    elif(categorization=='DB5'):
        #strings defining categories
        cat_labels = ['DB5','Non-DB5']
        #find indexes
        db5_idx = np.where(['DB5' in cell for cell in cell_labels[:,1]])
        ndb5_idx = np.where(np.invert(['DB5' in cell for cell in cell_labels[:,1]]))
        #apply categories to list
        cat[db5_idx] = 0
        cat[ndb5_idx] = 1      
        
    elif(categorization=='DB6'):
        #strings defining categories
        cat_labels = ['DB6','Non-DB6']
        #find indexes
        db6_idx = np.where(['DB6' in cell for cell in cell_labels[:,1]])
        ndb6_idx = np.where(np.invert(['DB6' in cell for cell in cell_labels[:,1]]))
        #apply categories to list
        cat[db6_idx] = 0
        cat[ndb6_idx] = 1        
        
    elif(categorization=='FMB'):
        #strings defining categories
        cat_labels = ['FMB','Non-FMB']
        #find indexes
        FMB_idx = np.where(['FMB' in cell for cell in cell_labels[:,1]])
        nFMB_idx = np.where(np.invert(['FMB' in cell for cell in cell_labels[:,1]]))
        #apply categories to list
        cat[FMB_idx] = 0
        cat[nFMB_idx] = 1   
        
    elif(categorization=='IMB'):
        #strings defining categories
        cat_labels = ['IMB','Non-IMB']
        #find indexes
        IMB_idx = np.where(['IMB' in cell for cell in cell_labels[:,1]])
        nIMB_idx = np.where(np.invert(['IMB' in cell for cell in cell_labels[:,1]]))
        #apply categories to list
        cat[IMB_idx] = 0
        cat[nIMB_idx] = 1   
        
    elif(categorization=='BB'):
        #strings defining categories
        cat_labels = ['BB','Non-BB']
        #find indexes
        BB_idx = np.where(['BB' in cell for cell in cell_labels[:,1]])
        nBB_idx = np.where(np.invert(['BB' in cell for cell in cell_labels[:,1]]))
        #apply categories to list
        cat[BB_idx] = 0
        cat[nBB_idx] = 1  
        
    elif(categorization=='GBC'):
        #strings defining categories
        cat_labels = ['GBC','Non-GBC']
        #find indexes
        GBC_idx = np.where(['GBC' in cell for cell in cell_labels[:,1]])
        nGBC_idx = np.where(np.invert(['GBC' in cell for cell in cell_labels[:,1]]))
        #apply categories to list
        cat[GBC_idx] = 0
        cat[nGBC_idx] = 1  
        
    else:
        print(f'{categorization} is not a valid cateogorization!')
        raise ValueError
    
    #check we assigned everything to a category
    if(any(cat==-1)):
        print(f'Cells Uncategorized!{cell_labels[np.where(cat==-1)]}')
        raise ValueError
    
    return(cat, cat_labels)

##### Normalize Data #####

def gauss_smoothing(table, width, std):
    """
    Performs gaussian smothing on a given table
    params:
        table (np.array): Table upon which gaussian smoothing will be performed.
        width (int): Width of smoothing window
        std (float): Standard deviation
    returns:
        gaussmooth_table (array): Gaussian-smoothed table
    """
    gaussmooth_table = np.zeros_like(table)
    b = gaussian(width, std)
    for i, trace in enumerate(table):
        smoothed_trace = filters.convolve1d(trace, b/b.sum())
        gaussmooth_table[i] = smoothed_trace
    return gaussmooth_table
            
def deriv_calc(table, absval=True):
    """
    Takes derivative of table
    params:
        table (array): table of data
    returns:
        dt_table (array): Derivative of table (NOT derivative multiplied by table)
    """
    dt_table = np.zeros_like(table)
    for i, trace in enumerate(table):
        t0 = trace[:-1]
        t1 = trace[1:]
        if absval == True:
            dt = np.abs(t1 - t0)
        else:
            dt = t1 - t0
        dt = np.append(0,dt)
        dt_table[i, :]=dt
    return dt_table

def scale_dt(table, gauss_sd=800):
    '''
    Scale each trace by the Gaussian Blurred dt
    Params:
        table (2d array): array of traces to be scaled
        gauss_sd (float): standard deviation of gaussian to be applied to traces
    Returns:
        scaled_table (2 array) array same size as table with scaled traces
    '''
    print(f'Scaling {np.shape(table)[0]} traces:', end='')
    scaled_table = np.zeros_like(table)
    for i, trace in enumerate(table):
        dt = np.append(0,np.abs(trace[1:]-trace[:-1]))
        dt = gaussian_filter1d(dt,gauss_sd)
        scaled_table[i] = dt
        print('*',end='')
    scaled_table = np.multiply(scaled_table, table)
    print("Done!")
    return(scaled_table)

### Turn this into flag
def norm1(table):
    '''
    Normalize to max 1 and min 0 \
    ***NOT SURE IF THIS IS THE SAME AS APRIL IMPLEMENTED***
    '''
    normed_table = np.zeros_like(table)
    for i, trace in enumerate(table):
        trace = (trace - np.min(trace))
        normed_table[i]  = trace / np.max(trace)
    return(normed_table)

def norm_meandepol_1(table):
    '''
    Normalize to mean of max depolarization 1 and min 0 \
    '''
    normed_table = np.zeros_like(table)
    for i, trace in enumerate(table):
        trace = (trace - np.min(trace))
        normed_table[i]  = trace / np.mean(trace[163000:167000])
    return(normed_table)
###

##### PCA #####

def run_pca(table, n_components):
    '''
    ##This function is broken!##
    Runs PCA on a given table for a given number of components
    Params:
        table (2d array): array of traces 
        n_components (int): Number of components to be visualized
    Returns:
        covar_matrix: Covariance matrix
        variance (list): list of variances
        components (list): list of components 
    '''
    #calculate variance explained and cumulative variance explained
    covar_matrix=sklearnPCA(n_components=10)
    covar_matrix.fit(table)
    variance=covar_matrix.explained_variance_ratio_
    var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
    
    #print graph of the variance explained with [n] features
    plt.ylabel('%variance explained')
    plt.xlabel('# of principle components')
    plt.title('Variance Explained')
   ## plt.ylim(70,100.5)
    plt.style.context('seaborn-whitegrid')
   # plt.plot(variance[:n_components])
    variance=covar_matrix.explained_variance_ratio_
    components=covar_matrix.components_
    return covar_matrix
    return variance
    return components
    
def pca_denoise(table, n_components, compare=False):
    '''
    Takes a table and performs PCA denoising on it. 
    Params:
        table (array): Array of traces
        n_components (int): Number of components to be subtracted
        compare: Flag for visualizations comparing original data to denoised data
    Returns:
        tables_r (array): Array of PCA-denoised traces
    '''
    #calculate variance explained and cumulative variance explained
    covar_matrix=sklearnPCA(n_components=10)
    covar_matrix.fit(table)
    variance=covar_matrix.explained_variance_ratio_
    var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
    
    #print graph of the variance explained with [n] features
    plt.ylabel('%variance explained')
    plt.xlabel('# of principle components')
    plt.title('Variance Explained')
   ## plt.ylim(70,100.5)
    plt.style.context('seaborn-whitegrid')
   # plt.plot(variance[:n_components])
    variance=covar_matrix.explained_variance_ratio_
    components=covar_matrix.components_
    return covar_matrix
    return variance
    return components
    tables_mean = np.mean(table,axis=0)
    #d_mean is also contained in sklearn_pca.mean_ after running

    #define our PCA object
    sklearn_pca = sklearnPCA(n_components)
    #fit our PCA object to our data
    sklearn_pca.fit(table)
    
    expl_var = sklearn_pca.explained_variance_ratio_
    plt.plot(expl_var[:n_components])
    plt.xlabel('Principle Component')
    plt.ylabel('Percentage of Explained Variance')
    plt.title('PCA Explained Variance by PC')
    plt.show()
    
    tf = sklearn_pca.components_.T
    print(f'TF shape: {tf.shape}')
    
    pf = (table - tables_mean)@tf
    print(f'PF shape: {pf.shape}')
    
    plt.plot(pf.T);
    plt.xlabel('Principle Component')
    plt.ylabel('value')
    plt.title('Time Series Data Projected in full PCA Space')
    ##plt.legend(np.arange(len(pf))+1)
    plt.show()
    
    tables_fullrecon = pf@(tf.T) + tables_mean
    ##
    
    plt.plot(table.T);
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title('Original Time Series Data')
    plt.show()

    plt.plot(tables_fullrecon.T);
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title('Reconstructed Time Series Data')
    
  ##  tables_fullrecon = pf@(tf.T) + tables_mean
    
    tr = tf[:,:n_components]
    print(f'TF shape: {tf.shape}')
    print(f'TR shape: {tr.shape}')

    pr = (table - tables_mean)@tr
    print(f'PR shape: {pr.shape}')
    
    plt.plot(pr.T);
    plt.xlabel('Principle Component')
    plt.ylabel('value')
    plt.title('Time Series Data Projected in Reduced PCA Space')
    ##plt.legend(np.arange(len(pf))+1)
    plt.show()
    
    tables_r = pr@(tr.T) + tables_mean
    print(f'tables_r shape: {tables_r.shape}')

    if compare == True:

        plt.plot(tables_fullrecon.flatten(), table.flatten(),'.')
        plt.xlabel('tables_a reconstructed')
        plt.ylabel('tables_a original')
        plt.title('Correlation between tables_a and tables_recon')
        plt.show()
        
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(tables_fullrecon[:,162000:].T);
        plt.xlabel('time')
        plt.ylabel('value')
        plt.title(f'Denoised Time Series Data: {n_components} Components Retained')
        plt.show()
        
        plt.subplot(1,2,2)
        plt.plot(table.T.flatten(), tables_fullrecon.T.flatten(),'.');
        plt.scatter(table.T.flatten(), tables_fullrecon.T.flatten())
        plt.xlim(-1,2)
        plt.ylim(-1,3)
        plt.ylabel('tables_a reconstructed')
        plt.xlabel('tables_a original')
        plt.title('Correlation between tables_a and tables_recon')
        plt.show()
        
        table_flattened = table.flatten()
        table_recon_flattened = tables_fullrecon.flatten()
        diff = table_flattened - table_recon_flattened
        plt.plot(table_flattened[::10], diff[::10], ".")
        plt.xlabel("tables_a voltage")
        plt.ylabel("difference in voltage")
        plt.show()
    return tables_r
    
def check_denoise(table, table_recon, tracenum):
    '''
    Visually compares individual traces from original and PCA-reconstructed data 
    Params:
        table (array): Original data
        table_recon (array): PCA-denoised data
        tracenum (int): Trace number (1-400) to be visualized
    '''
    #calculate variance explained and cumulative variance explained
    covar_matrix=sklearnPCA(n_components=10)
    covar_matrix.fit(table)
    variance=covar_matrix.explained_variance_ratio_
    var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
    
    #print graph of the variance explained with [n] features
    plt.ylabel('%variance explained')
    plt.xlabel('# of principle components')
    plt.title('Variance Explained')
   ## plt.ylim(70,100.5)
    plt.style.context('seaborn-whitegrid')
   # plt.plot(variance[:n_components])
    variance=covar_matrix.explained_variance_ratio_
    components=covar_matrix.components_
    return covar_matrix
    return variance
    return components
    diff_full = table - table_recon
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(np.asarray(table[tracenum,160000:]).T, label='original');
    plt.plot(np.asarray(table_recon[tracenum,160000:]).T, label='reconstructed');
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title('Both')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(np.asarray(diff_full[tracenum,160000:]).T);
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title('Difference')
    plt.show()
    return
    
def pcadenoise_visualizations(table, sweep16, allsweeps):
    '''
    Visualizes PCA-denoised data
    Params:
        table (2d array): array of traces 
        sweep16: Flag to visualize only sweep 16
        allsweeps: Flag to visualize all sweeps
    '''
    #calculate variance explained and cumulative variance explained
    covar_matrix=sklearnPCA(n_components=10)
    covar_matrix.fit(table)
    variance=covar_matrix.explained_variance_ratio_
    var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
    
    #print graph of the variance explained with [n] features
    plt.ylabel('%variance explained')
    plt.xlabel('# of principle components')
    plt.title('Variance Explained')
   ## plt.ylim(70,100.5)
    plt.style.context('seaborn-whitegrid')
   # plt.plot(variance[:n_components])
    variance=covar_matrix.explained_variance_ratio_
    components=covar_matrix.components_
    return covar_matrix
    return variance
    return components
    if sweep16 == True:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(table[:,162000:].T);
        plt.xlabel('time')
        plt.ylabel('value')
        plt.title(f'Original Time Series Data')
        plt.show()
        plt.figure(figsize=(10,5))
        
    if allsweeps == True: 
        plt.subplot(1,2,1)
        plt.plot(table.T);
        plt.xlabel('time')
        plt.ylabel('value')
        plt.title(f'Original Time Series Data')
        plt.show()

##### Kmeans clustering #####
        
def apply_kmeans(n_clusters, table):
    '''
    Apply the k-means algorithm to the traces in the table with clusternum clusters. Plot these results and return the cluster assignments if desired
    Params:
        clusternum (int): number of clusters desired
        table (2d array): rows are traces, cols are timeseries

    Returns:
        y_kmeans (int array): cluster assignments (only if return_clusters ==True)
    '''
    #run kmeans calculation
    kmeans = KMeans(n_clusters).fit(table)
    y_kmeans = kmeans.predict(table)
    #centers = kmeans.cluster_centers_
            
    return(y_kmeans)
            
def plot_clusters(plot_data, cluster_assignments, cats=None, cat_labels=None, lastsweep=True):
    '''
    Function to plot a set of traces assignged to cluster_assignments, with categories reflected.
    Parameters:
        plot_data (2d array): rows are traces, cols are timeseries
        cluster_assignments (1d array): Assignments to clusters for each trace in plot_data
        cats (int array): categorization for each cell
        cat_labels (label for each category type)
        lastsweep(bool): plot only the last sweep?
    '''
    #plotting colors for cluster assginments, or plot everything grey
    
    if cats is None:
        cats = np.zeros(np.shape(table)[0])
        cat_lables = [0]
   
    for i, c in enumerate(np.unique(cluster_assignments)):
        cells_list = np.where(cluster_assignments == i)[0]

        if lastsweep:
            #plt.figure(figsize=(8, 5))
            plt.figure()
            seen_labels = []
            for cell in cells_list:

                if cat_labels[cats[cell]] not in seen_labels:
                    seen_labels.append(cat_labels[cats[cell]])
                    label = cat_labels[cats[cell]]
                else:
                    label = None

                plt.plot(plot_data[cell,162000:], alpha=.5, color=f"C{cats[cell]}", label=label)
            plt.title(f"clustercenter {i}: {len(cells_list)} Sweep 16")
            plt.legend()
            plt.show()
        else: 
            #plt.figure(figsize=(8,5))
            plt.figure()
            seen_labels = []
            for cell in cells_list:

                if cat_labels[cats[cell]] not in seen_labels:
                    seen_labels.append(cat_labels[cats[cell]])
                    label = cat_labels[cats[cell]]
                else:
                    label = None

                plt.plot(plot_data[cell,:], alpha=.5, color=f"C{cats[cell]}", label=label)

            plt.plot(c)
            plt.title(f"clustercenter {i}: {len(cells_list)}")
            plt.legend()
            plt.show()
            
##### Agglomerative clustering #####

def agglom_clust(n_clusters, table):
    '''
    Performs agglomerative clustering on a table for a given number of clusters. 
    Params:
        table (2d array): array of traces
        n_clusters (int): Number of desired clusters. 
    Returns:
        clusterz (list): List of cluster numbers for each cell
    '''
    cluster = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward')  
    clusterz = cluster.fit_predict(table)
    
    return clusterz

##### Evaluate clusters #####
def homoscore(labels_true, labels_pred):
    homo_score = sklearn.metrics.homogeneity_score(labels_true, labels_pred)
    return homo_score

##### Silhouette plots #####

def silhouete_plots(table, range_clusters):
    '''
    Generates silhouete plots for a given range of clusters. 
    Params:
        table (2d array): array of traces 
        range_clusters (int range): Range of integer clusters
    Returns:
        Silhouete plots for each number of clusters
    '''
    
    print(__doc__)

    for n_clusters in range_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(table) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(table)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(table, cluster_labels)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(table, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
            "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
    plt.show()   



def silhouete_plots_agglom(table, range_clusters):
    '''
    Generates silhouete plots for agglomerative clustering
    Params:
        table (2d array): array of traces 
        range_clusters (range of ints): Range of cluster numbers to be visualized  
    '''
    #calculate variance explained and cumulative variance explained
    covar_matrix=sklearnPCA(n_components=10)
    covar_matrix.fit(table)
    variance=covar_matrix.explained_variance_ratio_
    var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
    
    #print graph of the variance explained with [n] features
    plt.ylabel('%variance explained')
    plt.xlabel('# of principle components')
    plt.title('Variance Explained')
   ## plt.ylim(70,100.5)
    plt.style.context('seaborn-whitegrid')
   # plt.plot(variance[:n_components])
    variance=covar_matrix.explained_variance_ratio_
    components=covar_matrix.components_
    return covar_matrix
    return variance
    return components
    print(__doc__)

    for n_clusters in range_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(table) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clustererr = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward')
        cluster_labelss = clustererr.fit_predict(table)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avgg = silhouette_score(table, cluster_labelss)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avgg)

        # Compute the silhouette scores for each sample
        sample_silhouette_valuess = silhouette_samples(table, cluster_labelss)

        yy_lower = 10
        for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
            ith_cluster_silhouette_valuess = \
                sample_silhouette_valuess[cluster_labelss == i]

            ith_cluster_silhouette_valuess.sort()

            size_cluster_ii = ith_cluster_silhouette_valuess.shape[0]
            yy_upper = yy_lower + size_cluster_ii

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(yy_lower, yy_upper), 0, ith_cluster_silhouette_valuess, facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, yy_lower + 0.5 * size_cluster_ii, str(i))

        # Compute the new y_lower for next plot
            yy_lower = yy_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avgg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
            "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
    plt.show()   
