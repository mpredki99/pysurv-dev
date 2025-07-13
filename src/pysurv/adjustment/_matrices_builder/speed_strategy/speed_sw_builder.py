import numpy as np


class SpeedSWBuilder:
    def __init__(self, parent) -> None:
        self._parent = parent
        
    def build_sw(self):
        sigmas = self._parent.dataset.controls.coordinates_sigma
        coord_indices = self._prepare_coord_indices()
        sW = self._parent.initialize_sw_matrix()
        
        self._fill_coord_sigmas(sigmas)
        sigmas = self._melt_coord_sigmas(sigmas)
        
        data = coord_indices.merge(
            sigmas,
            how="left",
            on=['id', 'coord'],
        )
        
        for row in data.itertuples(index=False):
            idx = getattr(row, "idx")
            sigma_value = getattr(row, "sigma_value")
            sW[idx] = 1 / (sigma_value ** 2) if sigma_value > 0 else 0
        return sW
    
    def _prepare_coord_indices(self):
        coord_columns = self._parent.dataset.controls.coordinates_columns
        indices = self._parent.cordinates_index_in_x_matrix.reset_index()
        indices = indices.melt(
            id_vars="id",
            value_vars=coord_columns, 
            var_name='coord', 
            value_name='idx'
        )
        return indices[indices["idx"] != -1]
    
    def _fill_coord_sigmas(self, sigmas):
        for col in  self._parent.dataset.controls.coordinates_columns:
            sigma_col = f"s{col}"
            if sigma_col in sigmas:
                sigmas[col] = sigmas[sigma_col].fillna(self._parent.default_sigmas[sigma_col])
                sigmas.drop(columns=sigma_col, inplace=True)
            else:
                sigmas[col] = self._parent.default_sigmas[sigma_col]
                
    def _melt_coord_sigmas(self, sigmas):
        sigmas = sigmas.reset_index()
        sigmas = sigmas.melt(
            id_vars="id",
            value_vars=sigmas.columns, 
            var_name='coord', 
            value_name='sigma_value'
        )
        return sigmas
