import torch
from utils.torch_funcs.function_LinearTriangle import get_triangular_nodes
from model.ParentClasses.class_ParameterToField import ParameterToField

class AnalyticalConstantBasisFunction(ParameterToField):
    def __init__(self, options) -> None:
        self.bf_grid = options["bf_grid"]
        nodes, nodesMap = get_triangular_nodes(self.bf_grid)
        self.center_points = torch.mean(torch.stack(nodes)[:,:2,:], dim=2).T
        n_bfs = len(nodes)
        super().__init__(options["s_grid"], n_bfs, options["bf_args"], options["BC_mask"])

        # create the coordinates of the triangles 
        nodes, nodesMap = get_triangular_nodes(self.bf_grid)
        self.center_points = torch.mean(torch.stack(nodes)[:,:2,:], dim=2).T
        
    def eval(self, x):
        x = self.apply_know_values(x)
        return x
    
    def set_given_in_rectangle(self, x_true, rectangle_options):
        mask1 = self.center_points[0, :] > rectangle_options["x_min"]
        mask2 = self.center_points[0, :] < rectangle_options["x_max"]
        mask3 = self.center_points[1, :] > rectangle_options["y_min"]
        mask4 = self.center_points[1, :] < rectangle_options["y_max"]
        mask = mask1 & mask2 & mask3 & mask4
        self.mask_given_values = mask
        self.given_value = x_true
        self.flag_given_values = True
        self.num_given_values = torch.sum(self.mask_given_values)
        self.num_unknowns = self.center_points.size(1) - self.num_given_values # number of unknowns

    def _create_set_of_bf(self):
        return None
    
    def _create_set_of_bf_with_BC(self):
        return None
    
    def _create_set_of_d_bf_with_BC__d_s_x(self):
        return None
    
    def _create_set_of_d_bf_with_BC__d_s_y(self):
        return None
    