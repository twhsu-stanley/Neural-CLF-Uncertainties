import torch
import numpy as np

torch.set_default_dtype(torch.float64)

__all__ = ['predict_tensor']

def predict_tensor(x, u, feature_names, coefficients):
        """ Compute the model predciction using expressions for torch tensor operations"""
        # x: array or tensor (batch_size x n_states)
        # u: array or tensor (batch_size x n_controls)
        # feature_names: list (len = n_features)
        # coefficients: array (size = n_states x n_features)
        
        n_features = len(feature_names)
        n_states = coefficients.shape[0]
        n_controls = u.shape[1]
        batch_size = x.shape[0]

        for s in range(n_states):
            locals()[f'x{s}'] = x[:,s]

        for s in range(n_controls):
            locals()[f'u{s}'] = u[:,s]
        
        #for i in range(n_features):
        #    feature_names[i] = feature_names[i].replace(" ", "*")
        #    feature_names[i] = feature_names[i].replace("^", "**")
        #    feature_names[i] = feature_names[i].replace("sin", "torch.sin")
        #    feature_names[i] = feature_names[i].replace("cos", "torch.cos")

        #f = np.zeros((batch_size, n_states))
        f = torch.zeros((batch_size, n_states), dtype = torch.float64)
        for s in range(n_states):
            for i in range(n_features):
                #f[:,s] = torch.add(f[:,s], torch.mul(eval(feature_names[i]), coefficients[s,i]))
                f[:,s] += eval(feature_names[i]) * coefficients[s,i]
        return f