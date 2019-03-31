path_full_set = "../data/dataset_complete.csv"
path_indices = "../data/indices.json"

loss_options = ['mean_squared_error', 'mean_absolute_error', 
                'mean_absolute_percentage_error', 'mean_squared_logarithmic_error']

cols = ['P.CH3 (wt%)', 'P.CH2 (wt%)', 'P.CH (wt%)', 'Olef (wt%)', 'Naph (wt%)',
        'Arom (wt%)', 'OH (wt%)', 'O (wt%)', 'Mol Wt', 'B.I', 'DCN']

col_y = 'DCN'

test_size = 0.15
