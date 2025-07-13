class MemorySWBuilder:
    def __init__(self, parent):
        self._parent = parent
        
    def build_sw(self):
        coord_indices = self._parent.cordinates_index_in_x_matrix
        sW = self._parent.initialize_sw_matrix()
        
        for idx in coord_indices.index:
            for coord in coord_indices.columns:
                sw_idx = coord_indices.at[idx, coord]
                if sw_idx == -1:
                    continue
                sigma_value = self._get_coord_sigma_value(idx, f"s{coord}")
                sW[sw_idx] = 1 / (sigma_value ** 2)
        
        return sW
                
    def _get_coord_sigma_value(self, idx, scoord):
        if scoord in self._parent.dataset.controls.columns:
            return self._parent.dataset.controls.at[idx, scoord]
        else:
            return self._parent.default_sigmas[scoord]
        