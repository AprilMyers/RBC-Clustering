## load in cell labels
import csv

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
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            cell_labels.append(np.array([row['Cell names'], row['Type'], row['Cut?']]))
    cell_labels = np.array(cell_labels)
    return(cell_labels)