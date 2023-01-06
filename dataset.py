from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class TablarDataset(Dataset):
    ATTRCONVD = {
        #'product_code': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4},
        'attribute_0': {'material_5': 0, 'material_7': 1},
        'attribute_1': {'material_5': 0, 'material_6': 1, 'material_7': 2, 'material_8': 3}
    }
    ATTRREVD = {
        #'product_code': ['A', 'B', 'C', 'D', 'E'],
        'attribute_0': ['material_5', 'material_7'],
        'attribute_1': ['meterial_5', 'material_6', 'material_7', 'material_8']
    }
    def __init__(self, df, x_only=False, drop_columns=['product_code', 'id', 'attribute_1']):
        self.x_only = x_only
        for dc in drop_columns:
            if dc in df.columns:
                df = df.drop(labels=[dc], axis='columns')

        if not x_only:
            self.ys = np.float32( df['failure'].values )

        for k, vd in TablarDataset.ATTRCONVD.items():
            if k in df.columns:
                df[k] = df[k].apply(lambda x: vd[x])

        # fill nan cell with 0 and add an attribute to indicate that
        nan_columns = df.columns[df.isna().any()].tolist()
        for col in nan_columns:
            df[f"{col}_isnan"] = df[col].isna()
            df[col].fillna(value=0, inplace=True)

        if x_only:
            self.xs = np.float32(df.values)
        else:
            self.xs = np.float32( df if not 'failure' in df.columns else df.drop(labels=['failure'], axis="columns").values )
            
        sc = StandardScaler()
        self.xs = sc.fit_transform(self.xs)
        self.in_features = self.xs.shape[1]

    def __getitem__(self, index):
        return (self.xs[index], self.ys[index]) if not self.x_only else self.xs[index]
    
    def __len__(self):
        return len(self.xs)
    