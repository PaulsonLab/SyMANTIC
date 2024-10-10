#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 23:24:41 2024

@author: muthyala.7
"""
import torch

import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd 

import pdb

class pareto:
    
    def __init__(self,rmse,complexity,final_pareto='no'):
        
        self.rmse = rmse
        
        self.complexity = abs(complexity)
        
        self.final_pareto = final_pareto
        
        

    def pareto_front(self):
        
        
        def is_pareto_efficient(costs):
            n_points = costs.shape[0]
            is_efficient = torch.ones(n_points, dtype=torch.bool)
            for i in range(n_points):
                if is_efficient[i]:
                    # Broadcasting comparison: compare all points to the i-th point
                    is_efficient &= torch.any(costs < costs[i], dim=1) | torch.all(costs == costs[i], dim=1)
                    is_efficient[i] = True  # Keep self
            return is_efficient
    
        def euclidean_distance(point1, point2):
            return torch.sqrt(torch.sum((point1 - point2) ** 2))
    
        # Combine RMSE and complexity into a single tensor
        costs = torch.column_stack((self.complexity, self.rmse))
    
        # Find the Pareto efficient points
        pareto_efficient_mask = is_pareto_efficient(costs)
        pareto_front = costs[pareto_efficient_mask]
        
        
    
        # Determine the utopia point (best possible values for each objective)
        utopia_point = torch.min(costs, axis=0).values
        #print(utopia_point)
        utopia_point = torch.tensor([0.0,0.0])
    
        # Calculate distances from utopia point for Pareto-efficient solutions
        distances = torch.tensor([euclidean_distance(point, utopia_point) for point in pareto_front])
    
        # Sort Pareto-efficient solutions by distance from utopia point
        sorted_indices = torch.argsort(distances)
        sorted_pareto_front = pareto_front[sorted_indices]
        sorted_distances = distances[sorted_indices]
        
        if self.final_pareto == 'yes':
            #pdb.set_trace()
            plt.figure(figsize=(10, 8))
            #plt.scatter(costs[:, 0], costs[:, 1], c='blue', label='All solutions')
            plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='red', label='Pareto front')
            sorted_pareto_front = pareto_front[pareto_front[:, 1].argsort()]
            plt.step(sorted_pareto_front[:, 0], sorted_pareto_front[:, 1], 'r-', where='pre', label='Pareto Line')
            plt.scatter(utopia_point[0], utopia_point[1], c='green', label='Utopia',marker='*',s=100)
            plt.xlabel(r'Complexity = $k \log n$ (bits)',weight='bold') 
            plt.ylabel('Accuracy (RMSE)',weight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.title('Pareto Frontier')
            plt.show()
        
        
            
    
        # Get the original indices of the Pareto efficient solutions, sorted by distance from utopia point
        pareto_indices = np.where(pareto_efficient_mask)[0]

        
        return pareto_indices