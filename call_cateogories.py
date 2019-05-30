import Utilities as utils

#load file list and tables_a
filelist = utils.load_filelist("/home/april/IActData_Export/*")
tables_a, final_flist = utils.generate_table(filelist, rm_cap=True, return_flist=True)
tables_a = utils.norm1(tables_a)

#load cell list
cell_labels = utils.readin_cell_labels('data/Cell_Types.csv')
if(np.alltrue(np.array(final_flist)==cell_labels[:,0])):
    print('Loaded {len(final_flist)} traces successfully!')
else:
    print('Warning! filelist does not match loaded in files! Neuron Assignments will be incorrect!!!')

#categorize and plot for DB
db_categorization, db_labels = utils.mark_cell_categories(cell_labels, 'DB')
utils.apply_kmeans(12, tables_a, db_categorization, db_labels)

#categorize and plot for onoff
onoff_categorization, onoff_labels = utils.mark_cell_categories(cell_labels, 'On_Off')
utils.apply_kmeans(12, tables_a, onoff_categorization, onoff_labels)