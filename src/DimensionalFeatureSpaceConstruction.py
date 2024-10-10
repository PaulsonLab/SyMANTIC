#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:17:36 2024

@author: muthyala.7
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:22:25 2024

@author: muthyala.7
"""

import torch

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import time

from itertools import combinations

import sys 

from scipy.stats import spearmanr

import pdb

from sympy import symbols,Pow,powdenest

from fractions import Fraction

import math

from .DimensionalRegressor import Regressor 

from .pareto_new import pareto

class feature_space_construction:
    
    '''
    Define the function to get the variables
    '''
    
    def __init__(self,df,operators=None,relational_units = None,initial_screening = None,no_of_operators=None,device='cpu',
                 dimensionality=None,metrics=[0.06,0.995],output_dim=None, test_x=None, test_y = None,test_variables=None,disp=False,pareto=False):
    
      '''
      ###########################################################################################
    
      no_of_operators - defines the presence of operators (binary or unary) in the expanded features space
    
      For example: if no_of_operators = 2 then the space will be limited to formation of features with 3 operators (x1+x2)/x3 or exp(x1+x2)
    
      ###########################################################################################
      '''
      self.no_of_operators = no_of_operators
    
      self.df = df
      
      self.disp = disp
      
      self.pareto = pareto
      
      '''
      ###########################################################################################
    
      operators [list type]: Defines the mathematical operators needs to be used in the feature expansion
    
      Please look at the README.md for type of mathematical operators allowed
    
      ###########################################################################################
      '''
      self.operators = operators
    
      self.operators_indexing = torch.arange(0,len(self.operators)).to(device)

    
      self.operators_dict = dict(zip(self.operators, self.operators_indexing.tolist()))    
      self.device = torch.device(device)
      
    
      # Filter the dataframe by removing the categorical datatypes and zero variance feature variables
      #if self.disp: ('###################  Removing the categorical variable columns, if there are any!!  ##################################################')
      
      self.df = self.df.select_dtypes(include=['float64','int64'])
      
      
      #Checking if we should go for the dimensionality 
      
      self .dimensionality = dimensionality
      
      #if len(self.dimensionality) != self.df.shape[1] - 1: sys.exit('Given dimensionality is not matching the number of features given.!!')
      
      if self.dimensionality !=None:
          
          #if self.disp: print('Extracting the dimensions of the same variables and will perform the feature expansion accordingly.... \n')
          
          self.relational_units = relational_units
          
    
      # Pop out the Targer variable of the problem and convert to tensor
      self.df.rename(columns = {f'{self.df.columns[0]}':'Target'},inplace=True)
      
      self.Target_column = torch.tensor(self.df.pop('Target')).to(self.device)
      # If initial screening is  yes then do the mic screening.....
      
      if initial_screening != None:
          
          self.screening = initial_screening[0]
          
          self.quantile = initial_screening[1]
          
          self.df, self.dimensionality = self.feature_space_screening(self.df, self.dimensionality)
          
          self.dimensionality = list(self.dimensionality)
          
          
          
    
      # Create the feature values tensor
      self.df_feature_values = torch.tensor(self.df.values).to(self.device)
      
      self.feature_names = self.df.columns.tolist()
      
      self.variables_indexing = torch.arange(0,len(self.feature_names)).reshape(1,-1).to(self.device)
      
      self.variables_dict = dict(zip(self.feature_names, self.variables_indexing.tolist()))
      
      self.dimensionality = symbols(self.dimensionality)
      
      self.rmse_metric = metrics[0]
      
      self.r2_metric = metrics[1]
      
      self.metrics = metrics
      
      self.output_dim = output_dim
      
      self.test_x = test_x
      self.test_y = test_y
      self.variable_names = test_variables
      
      self.pareto_points_identified = torch.empty(0,2).to(self.device)
      
      self.all_points_identified = torch.empty(0,2).to(self.device)
      
      self.p_exp =[]
      
      self.np_exp=[]
      
      self.operators_final = torch.empty(0,).to(self.device)
      
      self.operators_final = torch.full((self.df.shape[1],), float('nan')).to(self.device)
    
      self.variables_final = torch.empty(0,).to(self.device)
    
      self.reference_tensor = self.variables_indexing.clone().reshape(-1, 1)
      
      self.updated_pareto_rmse = torch.empty(0,).to(self.device)
      
      self.updated_pareto_r2 = torch.empty(0,).to(self.device)
      
      self.updated_pareto_complexity = torch.empty(0,).to(self.device)
      
      self.updated_pareto_names =[]
      
      self.update_pareto_coeff =torch.empty(0,).to(self.device)
      
      self.update_pareto_intercepts=torch.empty(0,).to(self.device)
      
      
      

      
      
    
    
    
    def get_dimensions_list(self):
        
        # Check for the shape of the feature variables and the length of the provided dimension
        
        if self.df_feature_values.shape[1] == len(self.dimensionality):
            print()
            #if self.disp: print('\n Shape of the dimension list and feature variable count matched... proceeding for further extraction and feature expansion.. \n')
            
        else:
            
            sys.exit('Mismatch between the dimension list provided and the number of feature variables... \n Please check the dimension list and feature variables and rerun the scipt.. \n ')
        
        #get the same dimensions from the list along with their index position.. 
        result ={}
        
        for index, value in enumerate(self.dimensionality):
            
            if value not in result:
                
                result[value] = []
                
            result[value].append(index)
        
        #if self.disp: print('Extraction of dimensionless and same dimension variables is completed!!.. \n')
        
        if symbols('1') in result.keys():
            
            self.dimension_less = result[symbols('1')]
            
            del result[symbols('1')]
            
            if self.disp: print(f'{len(self.dimension_less)} dimension less feature variables found in the given list!! \n')
        
            self.dimensions_index_dict = result
            
            del result
            
            
            return self.dimensions_index_dict, self.dimension_less
        
        else:
            
            self.dimensions_index_dict = result
            
            self.dimension_less = None
            
            return self.dimensions_index_dict,self.dimension_less
        
        
    def replace_strings_with_other_elements(self,target_strings,relational_units):

        # Function to find the other element for a single target string
        def find_other_element(target_string):
            
            found_tuple = next((tup for tup in relational_units if target_string in tup), None)
            
            if found_tuple:
                
                return found_tuple[1] if found_tuple[0] == target_string else found_tuple[1]
            
            return target_string  # Return the target string itself if no other element is found

        
        return [find_other_element(target_string) for target_string in target_strings]


    
    ### Cleaning the tensors
    def clean_tensor(self,tensor):
        

        
        mask = ~torch.isnan(tensor)
        
        
        counts = mask.sum(dim=1)
        
        
        max_count = counts.max()
        
        
        row_indices = torch.arange(tensor.shape[0]).unsqueeze(1).expand(-1, max_count)
        
        
        col_indices = torch.arange(max_count).unsqueeze(0).expand(tensor.shape[0], -1)
        
        
        valid_mask = col_indices < counts.unsqueeze(1)
        
        
        valid_elements = tensor[mask]
        
        
        result = torch.full((tensor.shape[0], max_count), float('nan'))
        
        
        result[row_indices[valid_mask], col_indices[valid_mask]] = valid_elements
        
        return result
    
    
    #### Feature expansion using dimension less numbers....
    
    def dimensionless_feature_expansion(self,iteration):
        
        feature_values_non_dimensional = torch.empty(self.df.shape[0],0).to(self.device)
        
        feature_names_non_dimensional =[]
        
        
        if self.dimension_less is None: 
            
            non_dimensions =[]
            
            feature_values_reference = torch.empty(0,).to(self.device)
            
            operators_reference = torch.empty(0,).to(self.device)
            
            #if self.disp: print('Non-Dimension feature expansion is skipped because of no non-dimension features....\n ')
            
            return feature_values_non_dimensional,feature_names_non_dimensional,non_dimensions
        
        
        non_dim_features = self.df_feature_values[:,self.dimension_less]
        
        non_dimensional_features = np.array(self.feature_names)[self.dimension_less]
        
        #if self.disp: print('##########################  Starting feature expansion of the non dimension feature variables... ################################################## \n')
        
        for op in self.operators:
            
            
            #Get the dimensionless variables
            reference_tensor1 = self.reference_tensor[self.dimension_less,:]
            
            try:
                #Extract the operators that are used in the variables (mainly useful for the expansions >1)
                operators_reference1 = self.operators_final[self.dimension_less,:]
                
            except:
                
                operators_reference1 = self.operators_final.unsqueeze(1)[self.dimension_less,:]
            
            ## Create an empty mutatable tensor and feature names list ####
            transformed_features = torch.empty(self.df.shape[0],0).to(self.device)
            
            transformed_feature_names = []
            
            feature_values_reference = torch.empty(0,).to(self.device)
            
            operators_reference = torch.empty(0,).to(self.device)
            
            #Transform the feature variables with exponential mathematical operator
            
            if op == 'exp':
                
                exp = torch.exp(non_dim_features)
                
                transformed_features = torch.cat((transformed_features,exp),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(exp('+ x + "))", non_dimensional_features)))
                
                if iteration == 1: 
                    
                    new_ref = reference_tensor1
                    
                    operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                
                else:
                    
                    new_ref = reference_tensor1
                    
                    if operators_reference1.shape[1] ==  iteration:
                    
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1
                    else:
                        
                        #operators_reference1 = operators_reference1.squeeze(1)
                        additional_columns = abs(operators_reference1.shape[1] - iteration)
                        
                        add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                        
                        operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                        
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                
                
                
            elif op == '/2':
                
                div2 = non_dim_features/2
                
                transformed_features = torch.cat((transformed_features,div2),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(('+ x + ")/2)", non_dimensional_features)))
                
                if iteration == 1: 
                    
                    new_ref = reference_tensor1
                    
                    operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                
                else:
                    
                    new_ref = reference_tensor1
                    
                    if operators_reference1.shape[1] ==  iteration:
                    
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1
                    else:
                        
                        #operators_reference1 = operators_reference1.squeeze(1)
                        additional_columns = abs(operators_reference1.shape[1] - iteration)
                        
                        add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                        
                        operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                        
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                
            elif op == '+1':
                
                div2 = non_dim_features +1
                
                transformed_features = torch.cat((transformed_features,div2),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(('+ x + "+1))", non_dimensional_features)))
                
                if iteration == 1: 
                    
                    new_ref = reference_tensor1
                    
                    operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                
                else:
                    
                    new_ref = reference_tensor1
                    
                    if operators_reference1.shape[1] ==  iteration:
                    
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1
                    else:
                        
                        #operators_reference1 = operators_reference1.squeeze(1)
                        additional_columns = abs(operators_reference1.shape[1] - iteration)
                        
                        add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                        
                        operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                        
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                
            elif op == '-1':
                
                div2 = non_dim_features -1
                
                transformed_features = torch.cat((transformed_features,div2),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(('+ x + "-1))", non_dimensional_features)))
                
                if iteration == 1: 
                    
                    new_ref = reference_tensor1
                    
                    operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                
                else:
                    
                    new_ref = reference_tensor1
                    
                    if operators_reference1.shape[1] ==  iteration:
                    
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1
                    else:
                        
                        #operators_reference1 = operators_reference1.squeeze(1)
                        additional_columns = abs(operators_reference1.shape[1] - iteration)
                        
                        add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                        
                        operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                        
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
            
            elif op == '/2pi':
                
                div2 = non_dim_features/(2*math.pi)
                
                transformed_features = torch.cat((transformed_features,div2),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(('+ x + "/2pi))", non_dimensional_features)))
                
                if iteration == 1: 
                    
                    new_ref = reference_tensor1
                    
                    operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                
                else:
                    
                    new_ref = reference_tensor1
                    
                    if operators_reference1.shape[1] ==  iteration:
                    
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1
                    else:
                        
                        #operators_reference1 = operators_reference1.squeeze(1)
                        additional_columns = abs(operators_reference1.shape[1] - iteration)
                        
                        add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                        
                        operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                        
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                
            elif op == '*2pi':
                
                div2 = non_dim_features*(2*math.pi)
                
                transformed_features = torch.cat((transformed_features,div2),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(('+ x + "*2pi))", non_dimensional_features)))
                
                if iteration == 1: 
                    
                    new_ref = reference_tensor1
                    
                    operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                
                else:
                    
                    new_ref = reference_tensor1
                    
                    if operators_reference1.shape[1] ==  iteration:
                    
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1
                    else:
                        
                        #operators_reference1 = operators_reference1.squeeze(1)
                        additional_columns = abs(operators_reference1.shape[1] - iteration)
                        
                        add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                        
                        operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                        
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)

            #Transform the feature variables with natural log mathematical operator
            
            elif op =='ln':
                
                ln = torch.log(non_dim_features)
                
                transformed_features = torch.cat((transformed_features,ln),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(ln('+x + "))", non_dimensional_features)))
                
                if iteration == 1: 
                    
                    new_ref = reference_tensor1
                    
                    operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                
                else:
                    
                    new_ref = reference_tensor1
                    
                    if operators_reference1.shape[1] ==  iteration:
                    
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1
                    else:
                        
                        #operators_reference1 = operators_reference1.squeeze(1)
                        additional_columns = abs(operators_reference1.shape[1] - iteration)
                        
                        add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                        
                        operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                        
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                
            
            #Transform the feature variables with log10 mathematical operator
            
            elif op =='log':
                
                log10 = torch.log10(non_dim_features)
                
                transformed_features = torch.cat((transformed_features,log10),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(log('+x + "))", non_dimensional_features)))
                
                if iteration == 1: 
                    
                    new_ref = reference_tensor1
                    
                    operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                
                else:
                    
                    new_ref = reference_tensor1
                    
                    if operators_reference1.shape[1] ==  iteration:
                    
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1
                    else:
                        
                        #operators_reference1 = operators_reference1.squeeze(1)
                        additional_columns = abs(operators_reference1.shape[1] - iteration)
                        
                        add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                        
                        operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                        
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                
            elif "pow" in op:
                
                import re
                
                pattern = r'\(([^)]*)\)'
                
                matches = re.findall(pattern, op)
                
                op = eval(matches[0])
                
                transformation = torch.pow(non_dim_features,op)
                
                transformed_features = torch.cat((transformed_features,transformation),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '('+x + f")**{matches[0]}", non_dimensional_features)))
                
                op = "pow(" + str(Fraction(op)) + ")"
                
                if iteration == 1: 
                    
                    new_ref = reference_tensor1
                    
                    operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                
                else:
                    
                    new_ref = reference_tensor1
                    
                    if operators_reference1.shape[1] ==  iteration:
                    
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1
                    else:
                        
                        #operators_reference1 = operators_reference1.squeeze(1)
                        additional_columns = abs(operators_reference1.shape[1] - iteration)
                        
                        add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                        
                        operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                        
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)

            #Transform the feature variables with SINE mathematical operator
            
            elif op =='sin':
                
                sin = torch.sin(non_dim_features)
                
                transformed_features = torch.cat((transformed_features,sin),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(sin('+x + "))", non_dimensional_features)))
                
                if iteration == 1: 
                    
                    new_ref = reference_tensor1
                    
                    operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                
                else:
                    
                    new_ref = reference_tensor1
                    
                    if operators_reference1.shape[1] ==  iteration:
                    
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1
                    else:
                        
                        #operators_reference1 = operators_reference1.squeeze(1)
                        additional_columns = abs(operators_reference1.shape[1] - iteration)
                        
                        add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                        
                        operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                        
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
             #Transform the feature variables with COSINE mathematical operator
             
            elif op =='cos':
                
                cos = torch.cos(non_dim_features)
                
                transformed_features = torch.cat((transformed_features,cos),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(cos('+x + "))", non_dimensional_features)))
                
                if iteration == 1: 
                    
                    new_ref = reference_tensor1
                    
                    operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                
                else:
                    
                    new_ref = reference_tensor1
                    
                    if operators_reference1.shape[1] ==  iteration:
                    
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1
                    else:
                        
                        #operators_reference1 = operators_reference1.squeeze(1)
                        additional_columns = abs(operators_reference1.shape[1] - iteration)
                        
                        add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                        
                        operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                        
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)

                
            #Transform the feature variables with reciprocal transformation
            
            elif op =='^-1':
                
                reciprocal = torch.reciprocal(non_dim_features)
                
                transformed_features = torch.cat((transformed_features,reciprocal),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(('+x + ")**-1)", non_dimensional_features)))
                
                if iteration == 1: 
                    
                    new_ref = reference_tensor1
                    
                    operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                
                else:
                    
                    new_ref = reference_tensor1
                    
                    if operators_reference1.shape[1] ==  iteration:
                    
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1
                    else:
                        
                        #operators_reference1 = operators_reference1.squeeze(1)
                        additional_columns = abs(operators_reference1.shape[1] - iteration)
                        
                        add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                        
                        operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                        
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                
            
            #Transform the feature variables with inverse exponential mathematical operator
            
            elif op =='exp(-1)':
                
                exp = torch.exp(non_dim_features)
                
                expreciprocal = torch.reciprocal(exp)
                
                transformed_features = torch.cat((transformed_features,expreciprocal),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(exp(-'+x + "))", non_dimensional_features)))
                
                if iteration == 1: 
                    
                    new_ref = reference_tensor1
                    
                    operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                
                else:
                    
                    new_ref = reference_tensor1
                    
                    if operators_reference1.shape[1] ==  iteration:
                    
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1
                    else:
                        
                        #operators_reference1 = operators_reference1.squeeze(1)
                        additional_columns = abs(operators_reference1.shape[1] - iteration)
                        
                        add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                        
                        operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                        
                        operators_reference1[:,-1] = self.operators_dict[op]
                        
                        operators_reference = operators_reference1

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                
                
            elif op == '+':
                
                if non_dim_features.shape[1] ==1: continue
                
                #generate the combinations on the fly
                combinations1 = list(combinations(non_dimensional_features,2))
                
                combinations2 = torch.combinations(torch.arange(non_dim_features.shape[1]),2)
                
                comb_tensor = non_dim_features.T[combinations2,:]
                
                comb_tensor = comb_tensor.permute(0,2,1)
                
                addition = torch.sum(comb_tensor,dim=2).T
                
                transformed_features = torch.cat((transformed_features,addition),dim=1)
                
                transformed_feature_names.extend(list(map(lambda comb: '('+'+'.join(comb)+')', combinations1)))
                
                del combinations1,comb_tensor
                
                new_ref = torch.cat([reference_tensor1[combinations2[:, 0]], 
                                   reference_tensor1[combinations2[:, 1]]], dim=1)
                
                max_cols = max(reference_tensor1.shape[1],new_ref.shape[1])
                
                self.reference_tensor = torch.cat([self.reference_tensor, torch.full((self.reference_tensor.size(0), max_cols - self.reference_tensor.size(1)), float('nan'))], dim=1)

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                

                if iteration == 1: 
                    
                    operators_reference = torch.full((new_ref.shape[0],), self.operators_dict[op])
                    
                else:
                    
                    op2 = operators_reference1[combinations2[:,0]]
                    
                    op3 = operators_reference1[combinations2[:,1]]
                    
                    op4 = torch.cat((op2,op3),dim=1)
                    
                    op4[:,-1] = self.operators_dict[op]


                    if self.operators_final.dim() ==1:  self.operators_final = self.operators_final.unsqueeze(1)
                    
                    nan_column = torch.full((self.operators_final.size(0), op4.shape[1] - self.operators_final.shape[1]), float('nan'))
                    
                    self.operators_final = torch.cat((self.operators_final, nan_column), dim=1)
                    
                    operators_reference = torch.cat((operators_reference, op4))
                    
                del combinations2
               
            elif op == '-':
                
                if non_dim_features.shape[1] ==1: continue
                
                combinations1 = list(combinations(non_dimensional_features,2))
                
                combinations2 = torch.combinations(torch.arange(non_dim_features.shape[1]),2)
                
                comb_tensor = non_dim_features.T[combinations2,:]
                
                comb_tensor = comb_tensor.permute(0,2,1)
                
                
                
                sub = torch.sub(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
                
                transformed_features = torch.cat((transformed_features,sub),dim=1)
                
                transformed_feature_names.extend(list(map(lambda comb: '('+'-'.join(comb)+')', combinations1)))
                
                del combinations1,comb_tensor
                
                new_ref = torch.cat([reference_tensor1[combinations2[:, 0]], 
                                   reference_tensor1[combinations2[:, 1]]], dim=1)
                
                max_cols = max(reference_tensor1.shape[1],new_ref.shape[1])
                
                self.reference_tensor = torch.cat([self.reference_tensor, torch.full((self.reference_tensor.size(0), max_cols - self.reference_tensor.size(1)), float('nan'))], dim=1)

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)

                if iteration == 1: 
                    
                    operators_reference = torch.full((new_ref.shape[0],), self.operators_dict[op])
                    
                else:
                    
                    op2 = operators_reference1[combinations2[:,0]]
                    
                    op3 = operators_reference1[combinations2[:,1]]
                    
                    op4 = torch.cat((op2,op3),dim=1)
                    
                    op4[:,-1] = self.operators_dict[op]


                    if self.operators_final.dim() ==1:  self.operators_final = self.operators_final.unsqueeze(1)
                    
                    nan_column = torch.full((self.operators_final.size(0), op4.shape[1] - self.operators_final.shape[1]), float('nan'))
                    
                    self.operators_final = torch.cat((self.operators_final, nan_column), dim=1)
                    
                    operators_reference = torch.cat((operators_reference, op4))
                
                del combinations2
                
            elif op == '*':
                
                if non_dim_features.shape[1] ==1: continue
                
                combinations1 = list(combinations(non_dimensional_features,2))
                
                combinations2 = torch.combinations(torch.arange(non_dim_features.shape[1]),2)
                
                comb_tensor = non_dim_features.T[combinations2,:]
                
                comb_tensor = comb_tensor.permute(0,2,1)

                mul = torch.multiply(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
                
                transformed_features = torch.cat((transformed_features,mul),dim=1)
                
                transformed_feature_names.extend(list(map(lambda comb: '('+'*'.join(comb)+')', combinations1)))
                
                del combinations1,comb_tensor
                
                new_ref = torch.cat([reference_tensor1[combinations2[:, 0]], 
                                   reference_tensor1[combinations2[:, 1]]], dim=1)
                
                max_cols = max(reference_tensor1.shape[1],new_ref.shape[1])
                
                self.reference_tensor = torch.cat([self.reference_tensor, torch.full((self.reference_tensor.size(0), max_cols - self.reference_tensor.size(1)), float('nan'))], dim=1)

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)

                if iteration == 1: 
                    
                    operators_reference = torch.full((new_ref.shape[0],), self.operators_dict[op])
                    
                else:
                    
                    op2 = operators_reference1[combinations2[:,0]]
                    
                    op3 = operators_reference1[combinations2[:,1]]
                    
                    op4 = torch.cat((op2,op3),dim=1)
                    
                    op4[:,-1] = self.operators_dict[op]


                    if self.operators_final.dim() ==1:  self.operators_final = self.operators_final.unsqueeze(1)
                    
                    nan_column = torch.full((self.operators_final.size(0), op4.shape[1] - self.operators_final.shape[1]), float('nan'))
                    
                    self.operators_final = torch.cat((self.operators_final, nan_column), dim=1)
                    
                    operators_reference = torch.cat((operators_reference, op4))
                del combinations2
                
            elif op == '/':
                
                if non_dim_features.shape[1] ==1: continue
                
                combinations1 = list(combinations(non_dimensional_features,2))
                
                combinations2 = torch.combinations(torch.arange(non_dim_features.shape[1]),2)
                
                comb_tensor = non_dim_features.T[combinations2,:]
                
                comb_tensor = comb_tensor.permute(0,2,1)
                
                div1 = torch.div(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
                
                div2 = torch.div(comb_tensor[:,:,1],comb_tensor[:,:,0]).T
                
                transformed_features = torch.cat((transformed_features,div1,div2),dim=1)
                
                transformed_feature_names.extend(list(map(lambda comb: '('+'/'.join(comb)+')', combinations1)))
                
                transformed_feature_names.extend(list(map(lambda comb: '('+'/'.join(comb[::-1])+')', combinations1)))
                
                del combinations1,comb_tensor
                
                new_ref = torch.cat([reference_tensor1[combinations2[:, 0]], 
                                   reference_tensor1[combinations2[:, 1]]], dim=1)
                
                max_cols = max(reference_tensor1.shape[1],new_ref.shape[1])
                
                self.reference_tensor = torch.cat([self.reference_tensor, torch.full((self.reference_tensor.size(0), max_cols - self.reference_tensor.size(1)), float('nan'))], dim=1)

                feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                
                feature_values_reference = feature_values_reference.repeat(2,1)
                
                if iteration == 1: 
                    
                    operators_reference = torch.full((new_ref.shape[0],1), self.operators_dict[op])
                    operators_reference = torch.cat((operators_reference,operators_reference))
                    
                else:
                    
                    op2 = operators_reference1[combinations2[:,0]]
                    
                    op3 = operators_reference1[combinations2[:,1]]
                    
                    op4 = torch.cat((op2,op3),dim=1)
                    
                    op4[:,-1] = self.operators_dict[op]


                    if self.operators_final.dim() ==1:  self.operators_final = self.operators_final.unsqueeze(1)
                    
                    nan_column = torch.full((self.operators_final.size(0), op4.shape[1] - self.operators_final.shape[1]), float('nan'))
                    
                    self.operators_final = torch.cat((self.operators_final, nan_column), dim=1)
                    
                    operators_reference = torch.cat((operators_reference, op4))
                    
                    operators_reference = torch.cat((operators_reference,operators_reference))
                del combinations2
                
                
            feature_values_non_dimensional = torch.cat((feature_values_non_dimensional,transformed_features),dim=1)
            
            feature_names_non_dimensional.extend(transformed_feature_names)
            
            
            self.reference_tensor = torch.cat((self.reference_tensor, feature_values_reference), dim=0)
            
            self.reference_tensor = self.clean_tensor(self.reference_tensor)
            
            if iteration >1:
                
                if self.operators_final.shape[1] == operators_reference.shape[1]:
                
                    self.operators_final = torch.cat((self.operators_final, operators_reference))
                    
                else:
                    
                    additional_columns = torch.full((self.operators_final.size(0), abs(operators_reference.shape[1]-self.operators_final.shape[1])), float('nan'))
                    
                    self.operators_final = torch.cat((self.operators_final,additional_columns),dim=1)
                    
                    self.operators_final = torch.cat((self.operators_final, operators_reference))
            else:
                if self.operators_final.dim() ==1: self.operators_final =  self.operators_final.unsqueeze(1)
                
                if operators_reference.dim() == 1: operators_reference =  operators_reference.unsqueeze(1)
                
                self.operators_final = torch.cat((self.operators_final, operators_reference))
                
            self.operators_final = self.clean_tensor(self.operators_final)

            # Check for the list of the features created whether it is empty or not and if it is empty return the empty tensors
         
        if len(feature_names_non_dimensional) == 0:
            
            non_dimensions=[]
            
            return feature_values_non_dimensional,feature_names_non_dimensional,non_dimensions
        
        # if feature names of non dimensional expansion is not zero then check for the nan and inf columns 
        
        else:
            '''
            nan_columns = torch.any(torch.isnan(feature_values_non_dimensional), dim=0)
            
            inf_columns = torch.any(torch.isinf(feature_values_non_dimensional), dim=0)
            
            nan_inf_columns = nan_columns|inf_columns
            
            feature_values_non_dimensional = feature_values_non_dimensional[:,~nan_inf_columns]
            
            feature_names_non_dimensional = [elem for i,elem in enumerate(feature_names_non_dimensional) if not nan_inf_columns[i]]
            '''
            non_dimensions = [symbols('1')]*feature_values_non_dimensional.shape[1]
            
            if self.disp: print('*********************** Completed the non dimensional feature expansion with features:', feature_values_non_dimensional.shape[1],'************************************************** \n')
            
            
            
            return feature_values_non_dimensional, feature_names_non_dimensional,non_dimensions
            
            
            
            
    def dimension_to_non_dimension_feature_expansion(self,iteration):
        
        #if self.disp: print('Starting the feature expansion for converting the dimensional to non dimensional features.....')
        
        # We are converting dimensional feature space to non-dimensional feature space
        dim_to_non_dim_feature_values = torch.empty(self.df.shape[0],0).to(self.device)
        
        dim_to_non_dim_feature_names=[]
        
        dim_to_non_dim_units =[]
            
        for dimension, batch in self.dimensions_index_dict.items():

            '''
            will perform the feature expansion converting the dimensional feature spaces to non dimensional feature space by applying operators like 
            
            exp, sin, cos, tan, log, ln, tan, tanh, sinh
            
            '''
            dim_features_values = self.df_feature_values[:,batch]
            
            dim_features_names = np.array(self.feature_names)[batch]
            
            reference_tensor1 = self.reference_tensor[batch,:]
            
            try:
                operators_reference1 = self.operators_final[batch,:]
                
            except:
                
                operators_reference1 = self.operators_final.unsqueeze(1)[batch,:]
            
            for op in self.operators:
                
                if op in ['+','*','-','/','/2','+1','-1','/2pi','*2pi','^-1']: continue
                
                if 'pow' in op: continue
                transformed_features = torch.empty(self.df.shape[0],0).to(self.device)
                
                transformed_feature_names = []
                
                feature_values_reference = torch.empty(0,).to(self.device)
                
                operators_reference = torch.empty(0,).to(self.device)
                
                if op == 'exp':
                    
                    exp = torch.exp(dim_features_values)
                    
                    transformed_features = torch.cat((transformed_features,exp),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(exp('+ x + "))", dim_features_names)))
                    
                    if iteration == 1: 
                        
                        new_ref = reference_tensor1
                        
                        operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                    
                    else:
                        
                        new_ref = reference_tensor1
                        
                        if operators_reference1.shape[1] ==  iteration:
                        
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1
                        else:
                            additional_columns = abs(operators_reference1.shape[1] - iteration)
                            
                            add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                            
                            operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                            
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                    
                    
                #Transform the feature variables with natural log mathematical operator
                
                elif op =='ln':
                    
                    ln = torch.log(dim_features_values)
                    
                    transformed_features = torch.cat((transformed_features,ln),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(ln('+x + "))", dim_features_names)))
                    
                    if iteration == 1: 
                        
                        new_ref = reference_tensor1
                        
                        operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                    
                    else:
                        
                        new_ref = reference_tensor1
                        
                        if operators_reference1.shape[1] ==  iteration:
                        
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1
                        else:
                            additional_columns = abs(operators_reference1.shape[1] - iteration)
                            
                            add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                            
                            operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                            
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                    
                #Transform the feature variables with log10 mathematical operator
                
                elif op =='log':
                    
                    log10 = torch.log10(dim_features_values)
                    
                    transformed_features = torch.cat((transformed_features,log10),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(log('+x + "))", dim_features_names)))
                    
                    if iteration == 1: 
                        
                        new_ref = reference_tensor1
                        
                        operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                    
                    else:
                        
                        new_ref = reference_tensor1
                        
                        if operators_reference1.shape[1] ==  iteration:
                        
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1
                        else:
                            additional_columns = abs(operators_reference1.shape[1] - iteration)
                            
                            add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                            
                            operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                            
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                    
                 
                elif op =='sin':
                    
                    
                    sin = torch.sin(dim_features_values)
                    
                    transformed_features = torch.cat((transformed_features,sin),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(sin('+x + "))", dim_features_names)))
                    
                    
                    if iteration == 1: 
                        
                        new_ref = reference_tensor1
                        
                        operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                    
                    else:
                        
                        new_ref = reference_tensor1
                        
                        if operators_reference1.shape[1] ==  iteration:
                        
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1
                        else:
                            additional_columns = abs(operators_reference1.shape[1] - iteration)
                            
                            add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                            
                            operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                            
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                    
                 #Transform the feature variables with COSINE mathematical operator
                 
                elif op =='cos':
                    
                    cos = torch.cos(dim_features_values)
                    
                    transformed_features = torch.cat((transformed_features,cos),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(cos('+x + "))", dim_features_names)))
                    
                    if iteration == 1: 
                        
                        new_ref = reference_tensor1
                        
                        operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                    
                    else:
                        
                        new_ref = reference_tensor1
                        
                        if operators_reference1.shape[1] ==  iteration:
                        
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1
                        else:
                            additional_columns = abs(operators_reference1.shape[1] - iteration)
                            
                            add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                            
                            operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                            
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                    
                 #Transform the feature variables with inverse exponential mathematical operator
                
                elif op =='exp(-1)':
                    
                    exp = torch.exp(dim_features_values)
                    
                    expreciprocal = torch.reciprocal(exp)
                    
                    transformed_features = torch.cat((transformed_features,expreciprocal),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(exp(-'+x + "))", dim_features_names)))
                    
                    if iteration == 1: 
                        
                        new_ref = reference_tensor1
                        
                        operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                    
                    else:
                        
                        new_ref = reference_tensor1
                        
                        if operators_reference1.shape[1] ==  iteration:
                        
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1
                        else:
                            additional_columns = abs(operators_reference1.shape[1] - iteration)
                            
                            add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                            
                            operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                            
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                    
                
                    
                    
                dim_to_non_dim_feature_values = torch.cat((dim_to_non_dim_feature_values,transformed_features),dim=1)
                
                dim_to_non_dim_feature_names.extend(transformed_feature_names)
                
                dim_to_non_dim_units = [symbols('1')]*dim_to_non_dim_feature_values.shape[1]
                
                
                self.reference_tensor = torch.cat((self.reference_tensor, feature_values_reference), dim=0)
                
                self.reference_tensor = self.clean_tensor(self.reference_tensor)
                
                
                if iteration >1:
        
                    
                    if self.operators_final.shape[1] == operators_reference.shape[1]:
                        
                        
                        self.operators_final = torch.cat((self.operators_final, operators_reference))
                        
                        self.operators_final = self.clean_tensor(self.operators_final)
                        
                    else:
                        
                        additional_columns = torch.full((self.operators_final.size(0), abs(self.operators_final.shape[1]-operators_reference.shape[1])), float('nan'))
                        
                        self.operators_final = torch.cat((self.operators_final,additional_columns),dim=1)
                        
                        self.operators_final = torch.cat((self.operators_final, operators_reference))
                        
                        self.operators_final = self.clean_tensor(self.operators_final)
                else:
                    
                    if self.operators_final.dim() ==1: self.operators_final =  self.operators_final.unsqueeze(1)
                    
                    if operators_reference.dim() == 1: operators_reference =  operators_reference.unsqueeze(1)
                    
                    
                    self.operators_final = torch.cat((self.operators_final, operators_reference))
                
                self.operators_final = self.clean_tensor(self.operators_final)
                
                
            
            
    
        '''        
        nan_columns = torch.any(torch.isnan(dim_to_non_dim_feature_values), dim=0)
        
        inf_columns = torch.any(torch.isinf(dim_to_non_dim_feature_values), dim=0)
        
        nan_inf_columns = nan_columns|inf_columns
        
        dim_to_non_dim_feature_values = dim_to_non_dim_feature_values[:,~nan_inf_columns]
        
        dim_to_non_dim_feature_names = [elem for i,elem in enumerate(dim_to_non_dim_feature_names) if not nan_inf_columns[i]]
        
        dim_to_non_dim_units = [symbols('1')]*dim_to_non_dim_feature_values.shape[1]
        '''
        
        if self.disp: print('********************************************* Dimension to nondimension feature expansion completed.... with feature space size:', dim_to_non_dim_feature_values.shape[1],'************************************************ \n')
        
        
        return dim_to_non_dim_feature_values, dim_to_non_dim_feature_names,dim_to_non_dim_units
        
        
    def dimension_feature_expansion(self,iteration):
        
        dimension_features_values = torch.empty(self.df.shape[0],0).to(self.device)
        
        dimension_features_names =[]
        
        dimension_values = [] #since we can't use tensors for strings we are going to add for the mul and other operators
        
        non_dimensional_div = torch.empty(self.df_feature_values.shape[0],0)
        
        non_dimensional_div_features = []
        
        non_dimensions_units=[]
        
        
        
        
        for dimension,batch in self.dimensions_index_dict.items():
            
            dim_features_values = self.df_feature_values[:,batch]
            
            dim_features_names = np.array(self.feature_names)[batch]
            
            dimension_copy = dimension
            
            reference_tensor1 = self.reference_tensor[batch,:]
            
            try:
                operators_reference1 = self.operators_final[batch,:]
                
            except:
                
                operators_reference1 = self.operators_final.unsqueeze(1)[batch,:]

            for op in self.operators:
                
                if op in ['exp','sin','cos','tanh','log','ln','exp(-1)']:continue
                
                feature_values_reference = torch.empty(0,).to(self.device)
                
                operators_reference = torch.empty(0,).to(self.device)
                
                
                transformed_features = torch.empty(self.df.shape[0],0).to(self.device)
                
                transformed_feature_names = []
                
                #pdb.set_trace()
                
                if op == '+':
                    
                    if len(dim_features_names) == 1: 
                        continue

                    
                    combinations1 = list(combinations(dim_features_names,2))
                    
                    combinations2 = torch.combinations(torch.arange(dim_features_values.shape[1]),2)
                    
                    comb_tensor = dim_features_values.T[combinations2,:]
                    
                    comb_tensor = comb_tensor.permute(0,2,1)
                    
                    addition = torch.sum(comb_tensor,dim=2).T
                    
                    transformed_features = torch.cat((transformed_features,addition),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda comb: '('+'+'.join(comb)+')', combinations1)))
                    
                    del addition,combinations1,comb_tensor
                    
                    dimensions_screened = [dimension]*transformed_features.shape[1]
                    
                    
                    if self.relational_units!=None:
                        
                        dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)

                    dimension_values.extend(dimensions_screened)
                    
                    new_ref = torch.cat([reference_tensor1[combinations2[:, 0]], 
                                       reference_tensor1[combinations2[:, 1]]], dim=1)
                    
                    max_cols = max(reference_tensor1.shape[1],new_ref.shape[1])
                    
                    self.reference_tensor = torch.cat([self.reference_tensor, torch.full((self.reference_tensor.size(0), max_cols - self.reference_tensor.size(1)), float('nan'))], dim=1)

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)

                    if iteration == 1: 
                        
                        operators_reference = torch.full((new_ref.shape[0],1), self.operators_dict[op])
                        
                    else:
                        
                        op2 = operators_reference1[combinations2[:,0]]
                        
                        op3 = operators_reference1[combinations2[:,1]]
                        
                        op4 = torch.cat((op2,op3),dim=1)
                        
                        op4[:,-1] = self.operators_dict[op]


                        if self.operators_final.dim() ==1:  self.operators_final = self.operators_final.unsqueeze(1)
                        
                        nan_column = torch.full((self.operators_final.size(0), op4.shape[1] - self.operators_final.shape[1]), float('nan'))
                        
                        self.operators_final = torch.cat((self.operators_final, nan_column), dim=1)
                        
                        operators_reference = torch.cat((operators_reference, op4))
                    
                elif op =='-':
                    
                    if len(dim_features_names) == 1: 
                        continue
                    
                    combinations1 = list(combinations(dim_features_names,2))
                    
                    combinations2 = torch.combinations(torch.arange(dim_features_values.shape[1]),2)
                    
                    comb_tensor = dim_features_values.T[combinations2,:]
                    
                    comb_tensor = comb_tensor.permute(0,2,1)
                    
                    sub = torch.sub(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
                    
                    transformed_features = torch.cat((transformed_features,sub),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda comb: '('+'-'.join(comb)+')', combinations1)))
                    
                    del combinations1,comb_tensor,sub
                    
                    dimensions_screened = [dimension]*transformed_features.shape[1]
                    
                    if self.relational_units!=None:
                        
                        dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                    
                    #add the dimension to the feature variables created
                    dimension_values.extend(dimensions_screened)
                    
                    new_ref = torch.cat([reference_tensor1[combinations2[:, 0]], 
                                       reference_tensor1[combinations2[:, 1]]], dim=1)
                    
                    max_cols = max(reference_tensor1.shape[1],new_ref.shape[1])
                    
                    self.reference_tensor = torch.cat([self.reference_tensor, torch.full((self.reference_tensor.size(0), max_cols - self.reference_tensor.size(1)), float('nan'))], dim=1)

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                    
                    #pdb.set_trace()
                    if iteration == 1: 
                        
                        operators_reference = torch.full((new_ref.shape[0],1), self.operators_dict[op])
                        
                    else:
                        
                        op2 = operators_reference1[combinations2[:,0]]
                        
                        op3 = operators_reference1[combinations2[:,1]]
                        
                        op4 = torch.cat((op2,op3),dim=1)
                        
                        op4[:,-1] = self.operators_dict[op]


                        if self.operators_final.dim() ==1:  self.operators_final = self.operators_final.unsqueeze(1)
                        
                        nan_column = torch.full((self.operators_final.size(0), op4.shape[1] - self.operators_final.shape[1]), float('nan'))
                        
                        self.operators_final = torch.cat((self.operators_final, nan_column), dim=1)
                        
                        operators_reference = torch.cat((operators_reference, op4))
                    del combinations2
                    
                elif op =='*':
                    
                    if len(dim_features_names) == 1: 
                        continue
                    
                    combinations1 = list(combinations(dim_features_names,2))
                    
                    combinations2 = torch.combinations(torch.arange(dim_features_values.shape[1]),2)
                    
                    comb_tensor = dim_features_values.T[combinations2,:]
                    
                    comb_tensor = comb_tensor.permute(0,2,1)
                    
                    mul = torch.multiply(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
                    
                    del comb_tensor
                    
                    transformed_features = torch.cat((transformed_features,mul),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda comb: '('+'*'.join(comb)+')', combinations1)))
                    
                    del combinations1,mul
                    
                    dimension = Pow(dimension,2)
                    
                    dimensions_screened = [dimension]*transformed_features.shape[1]
                    
                    if self.relational_units!=None:
                        
                        dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                    
                    #add the dimension to the feature variables created
                    dimension_values.extend(dimensions_screened)
                    
                    dimension=dimension_copy
                    
                    new_ref = torch.cat([reference_tensor1[combinations2[:, 0]], 
                                       reference_tensor1[combinations2[:, 1]]], dim=1)
                    
                    max_cols = max(reference_tensor1.shape[1],new_ref.shape[1])
                    
                    self.reference_tensor = torch.cat([self.reference_tensor, torch.full((self.reference_tensor.size(0), max_cols - self.reference_tensor.size(1)), float('nan'))], dim=1)

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                    
                    #pdb.set_trace()
                    if iteration == 1: 
                        
                        operators_reference = torch.full((new_ref.shape[0],1), self.operators_dict[op])
                        
                    else:
                        
                        op2 = operators_reference1[combinations2[:,0]]
                        
                        op3 = operators_reference1[combinations2[:,1]]
                        
                        op4 = torch.cat((op2,op3),dim=1)
                        
                        op4[:,-1] = self.operators_dict[op]


                        if self.operators_final.dim() ==1:  self.operators_final = self.operators_final.unsqueeze(1)
                        
                        nan_column = torch.full((self.operators_final.size(0), op4.shape[1] - self.operators_final.shape[1]), float('nan'))
                        
                        self.operators_final = torch.cat((self.operators_final, nan_column), dim=1)
                        
                        operators_reference = torch.cat((operators_reference, op4))
                        
                    del combinations2
                
                elif "pow" in op:
                    
                    import re
                    
                    pattern = r'\(([^)]*)\)'
                    
                    matches = re.findall(pattern, op)
                    
                    op = eval(matches[0])
                    
                    transformation = torch.pow(dim_features_values,op)
                    
                    transformed_features = torch.cat((transformed_features,transformation),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '('+x + f")**{matches[0]}", dim_features_names)))

                    dimension = powdenest(Pow(dimension,op), force=True)

                    if len(dim_features_names) == 1: 
                        
                        dimension=[dimension]
                        
                        if self.relational_units!=None:
                            
                            dimension = self.replace_strings_with_other_elements(dimension, self.relational_units)
                            
                        dimension_values.extend(dimension)
                        
                    else:
                        
                        dimensions_screened = [dimension]*transformed_features.shape[1]

                        if self.relational_units!=None:
                        
                            dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                        
                        #add the dimension to the feature variables created
                        dimension_values.extend(dimensions_screened)
                        
                    dimension=dimension_copy
                    
                    op = "pow(" + str(Fraction(op)) + ")"
                    
                    
                    
                    if iteration == 1: 
                        
                        new_ref = reference_tensor1
                        
                        operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                    
                    else:
                        
                        new_ref = reference_tensor1
                        
                        if operators_reference1.shape[1] ==  iteration:
                        
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1
                        else:
                            
                            #operators_reference1 = operators_reference1.squeeze(1)
                            additional_columns = abs(operators_reference1.shape[1] - iteration)
                            
                            add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                            
                            operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                            
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                    
                    
                
                elif op =='+1':

                    sum1 = dim_features_values + 1
                    
                    transformed_features = torch.cat((transformed_features,sum1),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(('+x + "+1))", dim_features_names)))
                    
                    if len(dim_features_names) == 1: 
                        
                        dimension=[dimension]
                        
                        if self.relational_units!=None:
                            
                            dimension = self.replace_strings_with_other_elements(dimension, self.relational_units)
                            
                        dimension_values.extend(dimension)
                        
                    else:
                        
                        dimensions_screened = [dimension]*transformed_features.shape[1]

                        if self.relational_units!=None:
                            
                            dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                        
                        #add the dimension to the feature variables created
                        dimension_values.extend(dimensions_screened)
                        
                    dimension=dimension_copy
                    
                    if iteration == 1: 
                        
                        new_ref = reference_tensor1
                        
                        operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                    
                    else:
                        
                        new_ref = reference_tensor1
                        
                        if operators_reference1.shape[1] ==  iteration:
                        
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1
                        else:
                            
                            #operators_reference1 = operators_reference1.squeeze(1)
                            additional_columns = abs(operators_reference1.shape[1] - iteration)
                            
                            add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                            
                            operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                            
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                
                elif op =='-1':

                    sum1 = dim_features_values - 1
                    
                    transformed_features = torch.cat((transformed_features,sum1),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(('+x + "-1))", dim_features_names)))
                    
                    if len(dim_features_names) == 1: 
                        
                        dimension=[dimension]
                        
                        if self.relational_units!=None:
                            
                            dimension = self.replace_strings_with_other_elements(dimension, self.relational_units)
                            
                        dimension_values.extend(dimension)
                        
                    else:
                        
                        dimensions_screened = [dimension]*transformed_features.shape[1]
                        
                        
                        if self.relational_units!=None:
                            
                            dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                        
                        #add the dimension to the feature variables created
                        dimension_values.extend(dimensions_screened)
                        
                    dimension=dimension_copy
                    
                    if iteration == 1: 
                        
                        new_ref = reference_tensor1
                        
                        operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                    
                    else:
                        
                        new_ref = reference_tensor1
                        
                        if operators_reference1.shape[1] ==  iteration:
                        
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1
                        else:
                            
                            #operators_reference1 = operators_reference1.squeeze(1)
                            additional_columns = abs(operators_reference1.shape[1] - iteration)
                            
                            add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                            
                            operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                            
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                    
                elif op =='/2':

                    sum1 = dim_features_values/2
                    
                    transformed_features = torch.cat((transformed_features,sum1),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(('+x + "/2))", dim_features_names)))
                    
                    if len(dim_features_names) == 1: 
                        
                        dimension=[dimension]
                        
                        if self.relational_units!=None:
                            
                            dimension = self.replace_strings_with_other_elements(dimension, self.relational_units)
                            
                        dimension_values.extend(dimension)
                        
                    else:
                        
                        dimensions_screened = [dimension]*transformed_features.shape[1]
                        
                        
                        if self.relational_units!=None:
                            
                            dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                        
                        #add the dimension to the feature variables created
                        dimension_values.extend(dimensions_screened)
                        
                    dimension=dimension_copy
                    
                    if iteration == 1: 
                        
                        new_ref = reference_tensor1
                        
                        operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                    
                    else:
                        
                        new_ref = reference_tensor1
                        
                        if operators_reference1.shape[1] ==  iteration:
                        
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1
                        else:
                            
                            #operators_reference1 = operators_reference1.squeeze(1)
                            additional_columns = abs(operators_reference1.shape[1] - iteration)
                            
                            add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                            
                            operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                            
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                
                elif op =='/2pi':

                    sum1 = dim_features_values/(2*math.pi)
                    
                    transformed_features = torch.cat((transformed_features,sum1),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(('+x + "/2pi))", dim_features_names)))
                    
                    
                    if len(dim_features_names) == 1: 
                        
                        dimension=[dimension]
                        
                        if self.relational_units!=None:
                            
                            dimension = self.replace_strings_with_other_elements(dimension, self.relational_units)
                            
                        dimension_values.extend(dimension)
                        
                    else:
                        
                        dimensions_screened = [dimension]*transformed_features.shape[1]
 
                        if self.relational_units!=None:
                            
                            dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                        
                        #add the dimension to the feature variables created
                        dimension_values.extend(dimensions_screened)
                        
                    dimension=dimension_copy
                    
                    if iteration == 1: 
                        
                        new_ref = reference_tensor1
                        
                        operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                    
                    else:
                        
                        new_ref = reference_tensor1
                        
                        if operators_reference1.shape[1] ==  iteration:
                        
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1
                        else:
                            
                            #operators_reference1 = operators_reference1.squeeze(1)
                            additional_columns = abs(operators_reference1.shape[1] - iteration)
                            
                            add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                            
                            operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                            
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                
                elif op =='*2pi':

                    sum1 = dim_features_values*(2*math.pi)
                    
                    transformed_features = torch.cat((transformed_features,sum1),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(('+x + "*2pi))", dim_features_names)))
                    
                    if len(dim_features_names) == 1: 
                        
                        dimension=[dimension]
                        
                        if self.relational_units!=None:
                            
                            dimension = self.replace_strings_with_other_elements(dimension, self.relational_units)
                            
                        dimension_values.extend(dimension)
                        
                    else:
                        
                        dimensions_screened = [dimension]*transformed_features.shape[1]

                        if self.relational_units!=None:
                            
                            dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                        
                        #add the dimension to the feature variables created
                        dimension_values.extend(dimensions_screened)
                        
                    dimension=dimension_copy
                    
                    if iteration == 1: 
                        
                        new_ref = reference_tensor1
                        
                        operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                    
                    else:
                        
                        new_ref = reference_tensor1
                        
                        if operators_reference1.shape[1] ==  iteration:
                        
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1
                        else:
                            
                            #operators_reference1 = operators_reference1.squeeze(1)
                            additional_columns = abs(operators_reference1.shape[1] - iteration)
                            
                            add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                            
                            operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                            
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
  
                    
                elif op =='^-1':

                    inverse = torch.pow(dim_features_values,-1)
                    
                    transformed_features = torch.cat((transformed_features,inverse),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(('+x + ")**-1)", dim_features_names)))
                    
                    dimension = Pow(dimension,-1)
                    
                    if len(dim_features_names) == 1: 
                        
                        dimension=[dimension]
                        
                        if self.relational_units!=None:
                            
                            dimension = self.replace_strings_with_other_elements(dimension, self.relational_units)
                            
                        dimension_values.extend(dimension)
                        
                    else:
                        
                        dimensions_screened = [dimension]*transformed_features.shape[1]
                        
                        if self.relational_units!=None:
                            
                            dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                        
                        #add the dimension to the feature variables created
                        dimension_values.extend(dimensions_screened)
                        
                    dimension=dimension_copy
                    
                    if iteration == 1: 
                        
                        new_ref = reference_tensor1
                        
                        operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                    
                    else:
                        
                        new_ref = reference_tensor1
                        
                        if operators_reference1.shape[1] ==  iteration:
                        
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1
                        else:
                            
                            #operators_reference1 = operators_reference1.squeeze(1)
                            additional_columns = abs(operators_reference1.shape[1] - iteration)
                            
                            add = torch.full((new_ref.size(0), additional_columns), float('nan'))
                            
                            operators_reference1  = torch.cat((operators_reference1, add),dim=1)
                            
                            operators_reference1[:,-1] = self.operators_dict[op]
                            
                            operators_reference = operators_reference1

                    feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
                    

                elif op =='/':
                    
                    if len(dim_features_names) == 1: 
                        continue
                    
                    combinations1 = list(combinations(dim_features_names,2))
                    
                    combinations2 = torch.combinations(torch.arange(dim_features_values.shape[1]),2)
                    
                    comb_tensor = dim_features_values.T[combinations2,:]
                    
                    comb_tensor = comb_tensor.permute(0,2,1)
                    
                    div1 = torch.div(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
                    
                    div2 = torch.div(comb_tensor[:,:,1],comb_tensor[:,:,0]).T
                    
                    transformed_features = torch.cat((transformed_features,div1,div2),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda comb: '('+'/'.join(comb)+')', combinations1)))
                    
                    transformed_feature_names.extend(list(map(lambda comb: '('+'/'.join(comb[::-1])+')', combinations1)))
                    
                    #pdb.set_trace()
                    del combinations1,comb_tensor,div1,div2
                    
                    new_ref = torch.cat([reference_tensor1[combinations2[:, 0]], 
                                       reference_tensor1[combinations2[:, 1]]], dim=1)
                    
                    max_cols = max(reference_tensor1.shape[1],new_ref.shape[1])
                    
                    self.reference_tensor = torch.cat([self.reference_tensor, torch.full((self.reference_tensor.size(0), max_cols - self.reference_tensor.size(1)), float('nan'))], dim=1)

                    feature_values_reference_div = torch.cat((feature_values_reference, new_ref), dim=0)
                    
                    
                    feature_values_reference_div = feature_values_reference_div.repeat(2,1)
                    
                    
                    if iteration == 1: 
                        
                        operators_reference_div = torch.full((new_ref.shape[0],1), self.operators_dict[op])
                        
                        operators_reference_div = torch.cat((operators_reference_div,operators_reference_div))
                        
                    else:
                        
                        op2 = operators_reference1[combinations2[:,0]]
                        
                        op3 = operators_reference1[combinations2[:,1]]
                        
                        op4 = torch.cat((op2,op3),dim=1)
                        
                        op4[:,-1] = self.operators_dict[op]


                        if self.operators_final.dim() ==1:  self.operators_final = self.operators_final.unsqueeze(1)
                        
                        nan_column = torch.full((self.operators_final.size(0), op4.shape[1] - self.operators_final.shape[1]), float('nan'))
                        
                        self.operators_final = torch.cat((self.operators_final, nan_column), dim=1)
                        
                        operators_reference_div = torch.cat((operators_reference, op4))
                        
                        operators_reference_div = torch.cat((operators_reference_div,operators_reference_div))
                        
                    del combinations2

                if op =='/':
                    
                    non_dimensional_div = torch.cat((non_dimensional_div,transformed_features),dim=1)
                    
                    non_dimensional_div_features.extend(transformed_feature_names)
                    
                    non_dimensions_units = [symbols('1')]*len(non_dimensional_div_features)
                    
                    self.reference_tensor = torch.cat((self.reference_tensor, feature_values_reference_div), dim=0)
                    
                    if iteration >1:
                        
                        if self.operators_final.shape[1] == operators_reference_div.shape[1]:
                        
                            self.operators_final = torch.cat((self.operators_final, operators_reference_div))
                            
                        else:
                            
                            additional_columns = torch.full((self.operators_final.size(0), abs(operators_reference_div.shape[1]-self.operators_final.shape[1])), float('nan'))
                            
                            self.operators_final = torch.cat((self.operators_final,additional_columns),dim=1)
                            
                            self.operators_final = torch.cat((self.operators_final, operators_reference_div))
                    else:
                        if self.operators_final.dim() ==1: self.operators_final =  self.operators_final.unsqueeze(1)
                        
                        if operators_reference_div.dim() == 1: operators_reference_div =  operators_reference_div.unsqueeze(1)
                        
                        self.operators_final = torch.cat((self.operators_final, operators_reference_div))
                    
                    '''
                    
                    nan_columns = torch.any(torch.isnan(non_dimensional_div), dim=0)
                    
                    inf_columns = torch.any(torch.isinf(non_dimensional_div), dim=0)
                    
                    nan_inf_columns = nan_columns|inf_columns
                    
                    non_dimensional_div = non_dimensional_div[:,~nan_inf_columns]
                    
                    non_dimensional_div_features = [elem for i,elem in enumerate(non_dimensional_div_features) if not nan_inf_columns[i]]
                    
                    non_dimensions_units = [elem for i,elem in enumerate(non_dimensions_units) if not nan_inf_columns[i]]
                    '''
                    
                else:
        
                    dimension_features_values = torch.cat((dimension_features_values,transformed_features),dim=1)
                    
                    dimension_features_names.extend(transformed_feature_names)
                    
                    try:
                    
                        self.reference_tensor = torch.cat((self.reference_tensor, feature_values_reference), dim=0)
                        
                        self.reference_tensor = self.clean_tensor(self.reference_tensor)
                        
                    except:
                        
                        
                        
                        additional_columns = torch.full((feature_values_reference.size(0), abs(feature_values_reference.shape[1]-self.reference_tensor.shape[1])), float('nan'))
                        
                        feature_values_reference = torch.cat((feature_values_reference,additional_columns),dim=1)
                        
                        self.reference_tensor = torch.cat((self.reference_tensor, feature_values_reference), dim=0)
                        
                        self.reference_tensor = self.clean_tensor(self.reference_tensor)
                        #pdb.set_trace()
                    
                    
                    if iteration >1:
                        
                        
                        if operators_reference.dim() == 1: operators_reference =  operators_reference.unsqueeze(1)
                        
                        if self.operators_final.shape[1] == operators_reference.shape[1]:
                        
                            self.operators_final = torch.cat((self.operators_final, operators_reference))
                            
                            self.operators_final = self.clean_tensor(self.operators_final)
                            
                        else:
                            
                            
                            additional_columns = torch.full((self.operators_final.size(0), abs(operators_reference.shape[1]-self.operators_final.shape[1])), float('nan'))
                            
                            
                            if operators_reference.shape[1] > self.operators_final.shape[1]:
                                
                                self.operators_final = torch.cat((self.operators_final,additional_columns),dim=1)
                                
                                self.operators_final = torch.cat((self.operators_final, operators_reference))
                                
                                self.operators_final = self.clean_tensor(self.operators_final)
                                
                            else:
                                
                                additional_columns = torch.full((operators_reference.size(0), abs(operators_reference.shape[1]-self.operators_final.shape[1])), float('nan'))
                                
                                operators_reference = torch.cat((operators_reference,additional_columns),dim=1)
                            
                                self.operators_final = torch.cat((self.operators_final, operators_reference))
                                
                                self.operators_final = self.clean_tensor(self.operators_final)
                        
                        
                    else:
                        
                        if self.operators_final.dim() ==1: self.operators_final =  self.operators_final.unsqueeze(1)
                        
                        if operators_reference.dim() == 1: operators_reference =  operators_reference.unsqueeze(1)
                        
                        self.operators_final = torch.cat((self.operators_final, operators_reference))
                        
                        self.operators_final = self.clean_tensor(self.operators_final)
                    

                    
                    #print('operators:',op,'featurenames:',len(dimension_features_names),'feature_values:',dimension_features_values.shape,'dimension_values:',len(dimension_values))
        '''    
        nan_columns = torch.any(torch.isnan(dimension_features_values), dim=0)
        
        inf_columns = torch.any(torch.isinf(dimension_features_values), dim=0)
        
        nan_inf_columns = nan_columns|inf_columns
        
        dimension_features_values = dimension_features_values[:,~nan_inf_columns]
        
        #pdb.set_trace()
        
        dimension_features_names = [elem for i,elem in enumerate(dimension_features_names) if not nan_inf_columns[i]]
        dimension_values = [elem for i,elem in enumerate(dimension_values) if not nan_inf_columns[i]]
        #pdb.set_trace()
        '''
        
        if self.disp: print('*********************************** Dimensional feature expansion completed.... with feature space size: ',dimension_features_values.shape[1],'************************************************** \n')
        
        return dimension_features_values, dimension_features_names, dimension_values,non_dimensional_div,non_dimensional_div_features,non_dimensions_units
    
    
    def inter_dimension_feature_expansion(self,iteration):
        
        combined_batch = torch.empty(self.df.shape[0],0)
        
        combined_dimensions =[]
        
        combined_feature_names=[]
        
        reference_tensor1 = torch.empty(0,)
        
        operators_reference1 = torch.empty(0,)
        
        
        
        for dimension,batch in self.dimensions_index_dict.items():
            
            combined_batch = torch.cat((combined_batch,self.df_feature_values[:,batch]),dim=1)
            
            combined_dimensions.extend([dimension]*len(batch))
            
            combined_feature_names.extend(np.array(self.feature_names)[batch])
            
            reference_tensor1 = torch.cat((reference_tensor1,self.reference_tensor[batch,:]))
            
            if self.operators_final.dim()==1: self.operators_final = self.operators_final.unsqueeze(1)
            
            operators_reference1 = torch.cat((operators_reference1,self.operators_final[batch,:]))
            
            
        if self.dimension_less !=None:    
            
            combined_batch = torch.cat((combined_batch,self.df_feature_values[:,self.dimension_less]),dim=1)
                
            combined_dimensions.extend([1]*len(self.dimension_less))
                
            combined_feature_names.extend(np.array(self.feature_names)[self.dimension_less])
            
            reference_tensor1 = torch.cat((reference_tensor1,self.reference_tensor[self.dimension_less,:]))
            
            if self.operators_final.dim()==1: self.operators_final = self.operators_final.unsqueeze(1)
            
            operators_reference1 = torch.cat((operators_reference1,self.operators_final[self.dimension_less,:]))
            
        
        
        # do the combinations and perform the multiplication and the division operations
        transformed_features = torch.empty(self.df.shape[0],0)
        
        transformed_feature_names=[]
        
        transformed_dimensions=[]
        
        
        if '*' in self.operators:
            
            
            
            feature_values_reference = torch.empty(0,)
            
            operators_reference = torch.empty(0,)
            
            combinations1 = list(combinations(combined_feature_names,2))
            
            combinations2 = torch.combinations(torch.arange(combined_batch.shape[1]),2)
            
            comb_tensor = combined_batch.T[combinations2,:]
            
            comb_tensor = comb_tensor.permute(0,2,1)
            
            mul = torch.multiply(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
            
            del comb_tensor
            
            #if iteration == 3: pdb.set_trace()
            
            transformed_features = torch.cat((transformed_features,mul),dim=1)
            
            #if iteration == 3: pdb.set_trace()
            
            transformed_feature_names.extend(['(' + '*'.join(comb) + ')' for comb in combinations1])
            
            #transformed_feature_names.extend(list(map(lambda comb: '('+'*'.join(comb)+')', combinations1)))
            
            del combinations1,mul
            
            combinations1 = list(combinations(combined_dimensions,2))
            
            #if iteration == 3: pdb.set_trace()
            

            transformed_dimensions.extend([x*y for x, y in combinations1])

            
            #process_tuple = lambda x, y: (Pow(x,2) if x == y 
            #                              else x*y)
            #transformed_dimensions.extend(list(map(lambda t: process_tuple(*t), combinations1)))
            
            #transformed_dimensions.extend([process_tuple(*t) for t in combinations1])

            
            del combinations1
            
            
            new_ref = torch.cat([reference_tensor1[combinations2[:, 0]], 
                               reference_tensor1[combinations2[:, 1]]], dim=1)
            
            max_cols = max(reference_tensor1.shape[1],new_ref.shape[1])
            
            self.reference_tensor = torch.cat([self.reference_tensor, torch.full((self.reference_tensor.size(0), max_cols - self.reference_tensor.size(1)), float('nan'))], dim=1)

            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
            
            
            
            if iteration == 1: 
                
                operators_reference = torch.full((new_ref.shape[0],1), self.operators_dict['*'])
                
                
            else:
                
                op2 = operators_reference1[combinations2[:,0]]
                
                op3 = operators_reference1[combinations2[:,1]]
                
                op4 = torch.cat((op2,op3),dim=1)
                
                op4[:,-1] = self.operators_dict['*']


                if self.operators_final.dim() ==1:  self.operators_final = self.operators_final.unsqueeze(1)
                
                nan_column = torch.full((self.operators_final.size(0), op4.shape[1] - self.operators_final.shape[1]), float('nan'))
                
                self.operators_final = torch.cat((self.operators_final, nan_column), dim=1)
                
                operators_reference = torch.cat((operators_reference, op4))
                
            self.reference_tensor = torch.cat((self.reference_tensor, feature_values_reference), dim=0)
            
            self.reference_tensor = self.clean_tensor(self.reference_tensor)
            
           
            if iteration >1:
                
                if self.operators_final.shape[1] == operators_reference.shape[1]:
                
                    self.operators_final = torch.cat((self.operators_final, operators_reference))
                    
                    self.operators_final = self.clean_tensor(self.operators_final)
                    
                else:
                    
                    additional_columns = torch.full((self.operators_final.size(0), abs(operators_reference.shape[1]-self.operators_final.shape[1])), float('nan'))
                    
                    self.operators_final = torch.cat((self.operators_final,additional_columns),dim=1)
                    
                    self.operators_final = torch.cat((self.operators_final, operators_reference))
                    
                    self.operators_final = self.clean_tensor(self.operators_final)
            else:
                if self.operators_final.dim() ==1: self.operators_final =  self.operators_final.unsqueeze(1)
                
                if operators_reference.dim() == 1: operators_reference =  operators_reference.unsqueeze(1)
                
                self.operators_final = torch.cat((self.operators_final, operators_reference))
                
                self.operators_final = self.clean_tensor(self.operators_final)
        
        
        if '/' in self.operators:
            
            #if iteration == 3: pdb.set_trace()
            
            feature_values_reference = torch.empty(0,)
            
            operators_reference = torch.empty(0,)
            
            combinations1 = list(combinations(combined_feature_names,2))
            
            combinations2 = torch.combinations(torch.arange(combined_batch.shape[1]),2)
            
            comb_tensor = combined_batch.T[combinations2,:]
            
            #del combinations2
            
            comb_tensor = comb_tensor.permute(0,2,1)

            div1 = torch.div(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
            
            div2 = torch.div(comb_tensor[:,:,1],comb_tensor[:,:,0]).T
            
            transformed_features = torch.cat((transformed_features,div1,div2),dim=1)
            
            #transformed_feature_names.extend(list(map(lambda comb: '('+'/'.join(comb)+')', combinations1)))
            
            transformed_feature_names.extend(['(' + '/'.join(comb) + ')' for comb in combinations1])
            
            transformed_feature_names.extend(['(' + '/'.join(comb[::-1]) + ')' for comb in combinations1])

            
            #transformed_feature_names.extend(list(map(lambda comb: '('+'/'.join(comb[::-1])+')', combinations1)))
            
            del combinations1
            
            combinations1 = list(combinations(combined_dimensions,2))
            
            dimensions=[]
            
            dimensions1=[]
            
            for x,y in combinations1:
                
                if x ==y: 
                    dimensions.append(symbols('1'))
                    
                    dimensions1.append(symbols('1'))
                else: 
                    dimensions.append(x/y)
                    
                    dimensions1.append(y/x)
    
            
            transformed_dimensions.extend(dimensions+dimensions1)

            del combinations1,dimensions,dimensions1
            
            new_ref = torch.cat([reference_tensor1[combinations2[:, 0]], 
                               reference_tensor1[combinations2[:, 1]]], dim=1)
            
            max_cols = max(reference_tensor1.shape[1],new_ref.shape[1])
            
            self.reference_tensor = torch.cat([self.reference_tensor, torch.full((self.reference_tensor.size(0), max_cols - self.reference_tensor.size(1)), float('nan'))], dim=1)

            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
            
            
            feature_values_reference = feature_values_reference.repeat(2,1)
            
            
            if iteration == 1: 
                
                operators_reference = torch.full((new_ref.shape[0],1), self.operators_dict['/'])
                
                operators_reference = torch.cat((operators_reference,operators_reference))
                
            else:
                
                op2 = operators_reference1[combinations2[:,0]]
                
                op3 = operators_reference1[combinations2[:,1]]
                
                op4 = torch.cat((op2,op3),dim=1)
                
                op4[:,-1] = self.operators_dict['/']


                if self.operators_final.dim() ==1:  self.operators_final = self.operators_final.unsqueeze(1)
                
                nan_column = torch.full((self.operators_final.size(0), op4.shape[1] - self.operators_final.shape[1]), float('nan'))
                
                self.operators_final = torch.cat((self.operators_final, nan_column), dim=1)
                
                operators_reference = torch.cat((operators_reference, op4))
                
                operators_reference = torch.cat((operators_reference,operators_reference))

            
            self.reference_tensor = torch.cat((self.reference_tensor, feature_values_reference), dim=0)
            
            self.reference_tensor = self.clean_tensor(self.reference_tensor)
            
            
            if iteration >1:
                
                if self.operators_final.shape[1] == operators_reference.shape[1]:
                
                    self.operators_final = torch.cat((self.operators_final, operators_reference))
                    
                    self.operators_final = self.clean_tensor(self.operators_final)
                    
                else:
                    
                    additional_columns = torch.full((self.operators_final.size(0), abs(operators_reference.shape[1]-self.operators_final.shape[1])), float('nan'))
                    
                    self.operators_final = torch.cat((self.operators_final,additional_columns),dim=1)
                    
                    self.operators_final = torch.cat((self.operators_final, operators_reference))
                    
                    self.operators_final = self.clean_tensor(self.operators_final)
            else:
                if self.operators_final.dim() ==1: self.operators_final =  self.operators_final.unsqueeze(1)
                
                if operators_reference.dim() == 1: operators_reference =  operators_reference.unsqueeze(1)
                
                self.operators_final = torch.cat((self.operators_final, operators_reference))
                
                self.operators_final = self.clean_tensor(self.operators_final)
            
        screened_dimensions = transformed_dimensions
        
        if self.relational_units!=None:
            
            screened_dimensions = self.replace_strings_with_other_elements(screened_dimensions, self.relational_units)
           
        
        transformed_dimensions = screened_dimensions

        if self.disp: print('**************************************** Inter dimensional feature expansion completed, with feature space size: ', transformed_features.shape[1],'*************************************************** \n')
        
        
        
        return transformed_features,transformed_feature_names,transformed_dimensions
        
    
    def feature_space_screening(self,df_sub,dimensions_screening):
        
        from sklearn.feature_selection import mutual_info_regression

        if self.screening == 'spearman':
            
            spear = spearmanr(df_sub.to_numpy(),self.Target_column,axis=0)
            
            screen1 = abs(spear.statistic)
            
            if screen1.ndim>1:screen1 = screen1[:-1,-1]
            
        elif self.screening=='mi':
            
            screen1 = mutual_info_regression(df_sub.to_numpy(), self.Target_column.numpy())
            
        
        
        df_screening = pd.DataFrame()
        
        df_screening['Feature variables'] = df_sub.columns
        
        df_screening['screen1'] = screen1
        
        df_screening = df_screening.sort_values(by = 'screen1',ascending= False).reset_index(drop=True)
        
        quantile_screen=df_screening.screen1.quantile(self.quantile)
        
        filtered_df = df_screening[(df_screening.screen1 > quantile_screen)].reset_index(drop=True)
        
        if filtered_df.shape[0]==0:
            filtered_df = df_screening[:int(df_sub.shape[1]/2)]

        df_screening1 = df_sub.loc[:,filtered_df['Feature variables'].tolist()]
        
        if len(dimensions_screening) == 0:
            
            return df_screening1,dimensions_screening
        
        indices = [df_sub.columns.tolist().index(item) for item in df_screening1.columns.tolist() if item in df_sub.columns.tolist()]

        screened_dimensions = np.array(dimensions_screening)[indices]
        
        
        return df_screening1, screened_dimensions
    
    
    def feature_expansion(self):
        
        if self.no_of_operators == None:
            
            #if self.disp: print('Implementing Autodepth and number of terms functionality...')
            
            
            
            i = 1
            
            start_time = time.time()
            
            # Get the dimension and non dimension variables... 
            
            self.get_dimensions_list()
            
            #Get the non dimension expansion... 
            
            
            
            non_dimension_feature_values, non_dimension_feature_names, non_dimension_units = self.dimensionless_feature_expansion(i)
            
            # Transform the dimension to non-dimension featur expansion....
            
            
            
            dim_to_non_dim_values,dim_to_non_dim_names,dim_to_non_dim_units = self.dimension_to_non_dimension_feature_expansion(i)
            
            
            
            #Transform the dimensional feature expansion... 
            
            dim_exp_feature_values, dim_exp_feature_names, dim_exp_units,non_dim_expanded_values,non_dim_expanded_names,non_dim_expanded_units = self.dimension_feature_expansion(i)
            
            # Inter-dimension feature expansion
            
            
            
            dim_inter_exp_values, dim_inter_exp_names,dim_inter_exp_units = self.inter_dimension_feature_expansion(i)
            
            # Concatenate the values to feature values, variables, dimensionality......
            
           
           
            self.df_feature_values = torch.cat((self.df_feature_values,non_dimension_feature_values,dim_to_non_dim_values,dim_exp_feature_values,dim_inter_exp_values,non_dim_expanded_values),dim=1)
        
            self.feature_names.extend(non_dimension_feature_names + dim_to_non_dim_names + dim_exp_feature_names + dim_inter_exp_names+non_dim_expanded_names)
            
            self.dimensionality.extend(list(non_dimension_units) + list(dim_to_non_dim_units) + list(dim_exp_units) + list(dim_inter_exp_units) + list(non_dim_expanded_units))
            

            end_time = time.time()
            
            if self.disp: print('***************************** Time taken for the initial feature expansion: ',end_time - start_time, ' seconds.. ****************************************** \n')
            
            if self.disp: print(f'***************************** Size of the feature space formed in the expansion {self.df_feature_values.shape[1]} ******************************** \n')
            
            # Replace NaNs with a value that won't interfere with counting
            tensor_replaced = torch.nan_to_num(self.reference_tensor, nan=float('inf'))

            # Mask to identify non-NaN values
            mask = self.reference_tensor == self.reference_tensor  # True where tensor is not NaN

            # Calculate the total number of numerical values per row
            num_numericals = mask.sum(dim=1)
            
            sorted_tensor, _ = torch.sort(self.reference_tensor, dim=1)
            
            diff = torch.diff(sorted_tensor, dim=1)
            
            unique_mask = torch.cat([torch.ones(self.reference_tensor.shape[0], 1).to(self.reference_tensor.device), (diff != 0).float()], dim=1)
            
            unique_counts = (unique_mask * mask).sum(dim=1)
            
            tensor_replaced1 = torch.nan_to_num(self.operators_final, nan=float('inf'))

            # Mask to identify non-NaN values
            mask = self.operators_final == self.operators_final  # True where tensor is not NaN

            # Calculate the total number of numerical values per row
            num_numericals1 = mask.sum(dim=1)
            
            sorted_tensor, _ = torch.sort(self.operators_final, dim=1)
            
            diff = torch.diff(sorted_tensor, dim=1)
            
            unique_mask = torch.cat([torch.ones(self.operators_final.shape[0], 1).to(self.operators_final.device), (diff != 0).float()], dim=1)

            unique_counts1 = (unique_mask * mask).sum(dim=1)
            
            complexity = (num_numericals+num_numericals1)*torch.log2(unique_counts+unique_counts1)
            
            complexity[:self.df.shape[1]] = 1
            
            self.dimension=None
            
            self.sis_features=None
            
            rmse1,equation1,r21,r,c,n,intercepts,coeffs,r2_value = Regressor(self.df_feature_values,self.Target_column,self.feature_names,self.dimensionality,complexity,metrics=self.metrics).regressor_fit()
            
            additional_columns = torch.full((1, abs(coeffs.shape[1])), float('nan'))
            
            additional_columns[:,0] = 1
            
            coeffs = torch.cat((additional_columns,coeffs))
            
            intercepts= torch.cat((torch.tensor([0]),intercepts))
            
            s= pareto(r,c).pareto_front()
            
            complexity_final = c[s]
            
            
            rmse_final = r[s]
            
            
            names_final = np.array(n)[s].tolist()
            
            r = rmse_final
            
            c= complexity_final
            
            n = names_final
            
            coeffs = coeffs[s]
            
            self.updated_pareto_rmse = torch.cat((self.updated_pareto_rmse,r))
            
            self.updated_pareto_r2 = torch.cat((self.updated_pareto_r2,r2_value[s]))
            
            self.updated_pareto_complexity = torch.cat((self.updated_pareto_complexity,c))
            
            self.updated_pareto_names.extend(n)
            
            if coeffs.dim()==1: coeffs = coeffs.unsqueeze(1)
            if self.update_pareto_coeff.dim()==1: self.update_pareto_coeff = self.update_pareto_coeff.unsqueeze(1)
            if coeffs.shape[1] == self.update_pareto_coeff.shape[1]:
                
                self.update_pareto_coeff = torch.cat((self.update_pareto_coeff,coeffs))
                
            else:
                if self.update_pareto_coeff.shape[1] < coeffs.shape[1]:
                    
                    additional_columns = torch.full((self.update_pareto_coeff.size(0), abs(coeffs.shape[1]-self.update_pareto_coeff.shape[1])), float('nan'))
                    
                    self.update_pareto_coeff = torch.cat((self.update_pareto_coeff,additional_columns),dim=1)
                    
                    self.update_pareto_coeff = torch.cat((self.update_pareto_coeff, coeffs))
                else:
                    additional_columns = torch.full((coeffs.size(0), abs(coeffs.shape[1]-self.update_pareto_coeff.shape[1])), float('nan'))
                    
                    coeffs = torch.cat((coeffs,additional_columns),dim=1)
                    
                    self.update_pareto_coeff = torch.cat((self.update_pareto_coeff, coeffs))
            
            self.update_pareto_intercepts=torch.cat((self.update_pareto_intercepts,intercepts[s]))
            
            
            
            if rmse1 <= self.rmse_metric and r21 >= self.r2_metric: 
                
                if self.pareto: final_pareto='yes'
                else: final_pareto = 'no'
                s= pareto(self.updated_pareto_rmse,self.updated_pareto_complexity,final_pareto=final_pareto).pareto_front()
                
                complexity_final = self.updated_pareto_complexity[s]
                
                
                rmse_final = self.updated_pareto_rmse[s]
                
                
                names_final = np.array(self.updated_pareto_names)[s].tolist()
                
                
                '''
                
                s= pn.pareto(r,c,final_pareto='yes').pareto_front()
                
                complexity_final = c[s]
                
                rmse_final = r[s]
                
                
                names_final = np.array(n)[s].tolist()
                '''
                
                intercepts = self.update_pareto_intercepts[s]
                
                coeffs = self.update_pareto_coeff[s]
                
                r2_final = self.updated_pareto_r2[s]
                
                
                data_final = {'Loss':rmse_final,'Complexity':complexity_final,'Equations':names_final,
                              'Intercepts':intercepts.tolist(),'Coefficients':coeffs.tolist(),'Score':r2_final}
                
                
                
                #data_final = {'Loss':rmse_final,'Complexity':complexity_final,'Equations':names_final}
                
                df_final = pd.DataFrame(data_final)
                
                df_unique = df_final.drop_duplicates(subset='Complexity')
                
                df_sorted = df_unique.sort_values(by='Complexity', ascending=True)
                
                df_sorted.reset_index(drop=True,inplace=True)
                
                #print('Equation:', equation1)
                
                return rmse1,equation1,r21,df_sorted 
            
            i = 2
            
            while True:
                
                if self.disp: print('\n',f'********************************* {i} Feature Expansion is Starting *********************************************** \n ')
                
                start_time = time.time()
                
                # Get the dimension and non dimension variables... 
                
                self.get_dimensions_list()
                
                #Get the non dimension expansion... 
                
                non_dimension_feature_values, non_dimension_feature_names, non_dimension_units = self.dimensionless_feature_expansion(i)
                
                # Transform the dimension to non-dimension featur expansion....
                
                dim_to_non_dim_values,dim_to_non_dim_names,dim_to_non_dim_units = self.dimension_to_non_dimension_feature_expansion(i)
                
                
                #Transform the dimensional feature expansion... 
                
                dim_exp_feature_values, dim_exp_feature_names, dim_exp_units,non_dim_expanded_values,non_dim_expanded_names,non_dim_expanded_units = self.dimension_feature_expansion(i)
                
                # Inter-dimension feature expansion
                
                dim_inter_exp_values, dim_inter_exp_names,dim_inter_exp_units = self.inter_dimension_feature_expansion(i)
                
                # Concatenate the values to feature values, variables, dimensionality......
               
                self.df_feature_values = torch.cat((self.df_feature_values,non_dimension_feature_values,dim_to_non_dim_values,dim_exp_feature_values,dim_inter_exp_values,non_dim_expanded_values),dim=1)
            
                self.feature_names.extend(non_dimension_feature_names + dim_to_non_dim_names + dim_exp_feature_names + dim_inter_exp_names+non_dim_expanded_names)
                
                self.dimensionality.extend(list(non_dimension_units) + list(dim_to_non_dim_units) + list(dim_exp_units) + list(dim_inter_exp_units) + list(non_dim_expanded_units))
                
                # Replace NaNs with a value that won't interfere with counting
                tensor_replaced = torch.nan_to_num(self.reference_tensor, nan=float('inf'))

                # Mask to identify non-NaN values
                mask = self.reference_tensor == self.reference_tensor  # True where tensor is not NaN

                # Calculate the total number of numerical values per row
                num_numericals = mask.sum(dim=1)
                
                sorted_tensor, _ = torch.sort(self.reference_tensor, dim=1)
                
                diff = torch.diff(sorted_tensor, dim=1)
                
                unique_mask = torch.cat([torch.ones(self.reference_tensor.shape[0], 1).to(self.reference_tensor.device), (diff != 0).float()], dim=1)
                
                unique_counts = (unique_mask * mask).sum(dim=1)
                
                tensor_replaced1 = torch.nan_to_num(self.operators_final, nan=float('inf'))

                # Mask to identify non-NaN values
                mask = self.operators_final == self.operators_final  # True where tensor is not NaN

                # Calculate the total number of numerical values per row
                num_numericals1 = mask.sum(dim=1)
                
                sorted_tensor, _ = torch.sort(self.operators_final, dim=1)
                
                diff = torch.diff(sorted_tensor, dim=1)
                
                unique_mask = torch.cat([torch.ones(self.operators_final.shape[0], 1).to(self.operators_final.device), (diff != 0).float()], dim=1)

                unique_counts1 = (unique_mask * mask).sum(dim=1)
                
                complexity = (num_numericals+num_numericals1)*torch.log2(unique_counts+unique_counts1)
                
                complexity[:self.df.shape[1]] = 1

                end_time = time.time()
                
                if self.disp: print('***************************************** Time taken for the initial feature expansion: ',end_time - start_time, ' seconds *********************************************************** \n')
                
                if self.disp: print('******************************************** Size of the feature space formed in the initial expansion',self.df_feature_values.shape[1],'************************************************ \n')
                
                rmse,equation,r2,r,c,n,intercepts,coeffs,r2_value = Regressor(self.df_feature_values,self.Target_column,self.feature_names,self.dimensionality,complexity,metrics=self.metrics).regressor_fit()
                
                additional_columns = torch.full((1, abs(coeffs.shape[1])), float('nan'))
                
                additional_columns[:,0] = 1
                
                coeffs = torch.cat((additional_columns,coeffs))
                
                intercepts= torch.cat((torch.tensor([0]),intercepts))
                
                
                s= pareto(r,c).pareto_front()
                
                complexity_final = c[s]
                
                rmse_final = r[s]
                
                
                names_final = np.array(n)[s].tolist()
                
                r = rmse_final
                
                c= complexity_final
                
                n = names_final
                
                coeffs = coeffs[s]
                
                
                self.updated_pareto_rmse = torch.cat((self.updated_pareto_rmse,r))
                
                self.updated_pareto_r2 = torch.cat((self.updated_pareto_r2,r2_value[s]))
                
                self.updated_pareto_complexity = torch.cat((self.updated_pareto_complexity,c))
                
                self.updated_pareto_names.extend(n)
                
               
                if coeffs.shape[1] == self.update_pareto_coeff.shape[1]:
                    
                    self.update_pareto_coeff = torch.cat((self.update_pareto_coeff,coeffs))
                    
                else:
                    if self.update_pareto_coeff.shape[1] < coeffs.shape[1]:
                        
                        additional_columns = torch.full((self.update_pareto_coeff.size(0), abs(coeffs.shape[1]-self.update_pareto_coeff.shape[1])), float('nan'))
                        
                        self.update_pareto_coeff = torch.cat((self.update_pareto_coeff,additional_columns),dim=1)
                        
                        self.update_pareto_coeff = torch.cat((self.update_pareto_coeff, coeffs))
                    else:
                        additional_columns = torch.full((coeffs.size(0), abs(coeffs.shape[1]-self.update_pareto_coeff.shape[1])), float('nan'))
                        
                        coeffs = torch.cat((coeffs,additional_columns),dim=1)
                        
                        self.update_pareto_coeff = torch.cat((self.update_pareto_coeff, coeffs))
                    
                
                self.update_pareto_intercepts=torch.cat((self.update_pareto_intercepts,intercepts[s]))
                
                
                if rmse <= self.rmse_metric and r2 >= self.r2_metric:
                    
                    break
                
                if i >=2 and self.df_feature_values.shape[1]>10000:
                    
                    if self.disp: print('Expanded feature space is::',self.df_feature_values.shape[1])
                    
                    if self.disp: print('!!Warning:: Further feature expansions result in huge memory consumption and may result in crashing the system...')
                    
                    response = input("Do you want to continue (yes/no)? ").strip().lower()
                    
                    if response == 'no': 
                        
                        if self.disp: print("Exiting based on user input.")
                        
                        break
                i = i+1
                
            
            
            if self.pareto: final_pareto = 'yes'
            else: final_pareto = 'no'
            
            s= pareto(self.updated_pareto_rmse,self.updated_pareto_complexity,final_pareto=final_pareto).pareto_front()
            
            complexity_final = self.updated_pareto_complexity[s]
            
            rmse_final = self.updated_pareto_rmse[s]
            
            
            names_final = np.array(self.updated_pareto_names)[s].tolist()
            
            intercepts = self.update_pareto_intercepts[s]
            
            coeffs = self.update_pareto_coeff[s]
            
            r2_final = self.updated_pareto_r2[s]
            
            data_final = {'Loss':rmse_final,'Complexity':complexity_final,'Equations':names_final,
                          'Intercepts':intercepts.tolist(),'Coefficients':coeffs.tolist(),'Score':r2_final}
            
            
            
            #data_final = {'Loss':rmse_final,'Complexity':complexity_final,'Equations':names_final}
            
            df_final = pd.DataFrame(data_final)
            
            df_unique = df_final.drop_duplicates(subset='Complexity')
            
            df_sorted = df_unique.sort_values(by='Complexity', ascending=True)
            
            df_sorted.reset_index(drop=True,inplace=True)
            
            #print('Equation:', equation)
            
            return rmse,equation,r2,df_sorted
        
        else:
            
            for i in range(1,self.no_of_operators):
            
                start_time = time.time()
                
                # Get the dimension and non dimension variables... 
                
                self.get_dimensions_list()
                
                #Get the non dimension expansion... 
                
                non_dimension_feature_values, non_dimension_feature_names, non_dimension_units = self.dimensionless_feature_expansion(i)
                
                
                # Transform the dimension to non-dimension featur expansion....
                
                dim_to_non_dim_values,dim_to_non_dim_names,dim_to_non_dim_units = self.dimension_to_non_dimension_feature_expansion(i)
                
                
                #Transform the dimensional feature expansion... 
                
                dim_exp_feature_values, dim_exp_feature_names, dim_exp_units,non_dim_expanded_values,non_dim_expanded_names,non_dim_expanded_units = self.dimension_feature_expansion(i)
                
                # Inter-dimension feature expansion
                
                dim_inter_exp_values, dim_inter_exp_names,dim_inter_exp_units = self.inter_dimension_feature_expansion(i)
                
                # Concatenate the values to feature values, variables, dimensionality......
               
                self.df_feature_values = torch.cat((self.df_feature_values,non_dimension_feature_values,dim_to_non_dim_values,dim_exp_feature_values,non_dim_expanded_values,dim_inter_exp_values),dim=1)
            
                self.feature_names.extend(non_dimension_feature_names + dim_to_non_dim_names + dim_exp_feature_names+non_dim_expanded_names + dim_inter_exp_names)
                
                self.dimensionality.extend(list(non_dimension_units) + list(dim_to_non_dim_units) + list(dim_exp_units) + list(non_dim_expanded_units) + list(dim_inter_exp_units))
                
    
                end_time = time.time()
                
                if self.disp: print(f'****************************************** Time taken for the {i} feature expansion: ',end_time - start_time, ' seconds ************************************* \n')
                
                if self.disp: print(f'******************************************** Size of the feature space formed in the {i} expansion',self.df_feature_values.shape[1],'********************************** \n')
                
                #pdb.set_trace()
                
                if self.operators_final.dim() !=2:
                
                    self.operators_final = self.operators_final.unsqueeze(1)
            
            # Replace NaNs with a value that won't interfere with counting
            tensor_replaced = torch.nan_to_num(self.reference_tensor, nan=float('inf'))
    
            # Mask to identify non-NaN values
            mask = self.reference_tensor == self.reference_tensor  # True where tensor is not NaN
    
            # Calculate the total number of numerical values per row
            num_numericals = mask.sum(dim=1)
            
            sorted_tensor, _ = torch.sort(self.reference_tensor, dim=1)
            
            diff = torch.diff(sorted_tensor, dim=1)
            
            unique_mask = torch.cat([torch.ones(self.reference_tensor.shape[0], 1).to(self.reference_tensor.device), (diff != 0).float()], dim=1)
            
            unique_counts = (unique_mask * mask).sum(dim=1)
            
            tensor_replaced1 = torch.nan_to_num(self.operators_final, nan=float('inf'))
    
            # Mask to identify non-NaN values
            mask = self.operators_final == self.operators_final  # True where tensor is not NaN
    
            # Calculate the total number of numerical values per row
            num_numericals1 = mask.sum(dim=1)
            
            sorted_tensor, _ = torch.sort(self.operators_final, dim=1)
            
            diff = torch.diff(sorted_tensor, dim=1)
            
            unique_mask = torch.cat([torch.ones(self.operators_final.shape[0], 1).to(self.operators_final.device), (diff != 0).float()], dim=1)
    
            unique_counts1 = (unique_mask * mask).sum(dim=1)
            
            complexity = (num_numericals+num_numericals1)*torch.log2(unique_counts+unique_counts1)
            
            complexity[:self.df.shape[1]] = 1
            
            return self.df_feature_values,self.Target_column,self.feature_names,self.dimensionality,complexity
    
        
            
        
    
    
    
    
           
         
    
