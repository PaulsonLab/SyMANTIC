'''
##############################################################################################

Importing the required libraries

##############################################################################################
'''
import torch

import pandas as pd

import numpy as np

import warnings

import itertools

import time

from sklearn.feature_selection import mutual_info_regression        

from scipy.stats import spearmanr

from itertools import combinations

import pdb

from fractions import Fraction

from .pareto_new import pareto

from .Regressor import Regressor


class feature_space_construction:

  '''
  ##############################################################################################################

  Define global variables like number of operators and the input data frame and the operator set given

  ##############################################################################################################
  '''
  def __init__(self,operators,df,no_of_operators=None,device='cpu',initial_screening=None,metrics=[0.06,0.995],disp=False,pareto=False,dimension=3,sis_features=20,feature_names=False):

    '''
    ###########################################################################################

    no_of_operators - defines the presence of operators (binary or unary) in the expanded features space

    For example: if no_of_operators = 2 then the space will be limited to formation of features with 3 operators (x1+x2)/x3 or exp(x1+x2)

    ###########################################################################################
    '''
    self.no_of_operators = no_of_operators

    self.df = df
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

    self.df = self.df.select_dtypes(include=['float64','int64','float32','int32'])
    
    # Compute the variance of each column
    variance = self.df.var()

    # Get the names of the zero variance columns
    zero_var_cols = variance[variance == 0].index

    # Drop the zero variance columns from the dataframe
    self.df = self.df.drop(zero_var_cols, axis=1)
    
    # Pop out the Targer variable of the problem and convert to tensor
    self.df.rename(columns = {f'{self.df.columns[0]}':'Target'},inplace=True)
    
    self.Target_column = torch.tensor(self.df.pop('Target')).to(self.device)
    
    
    if initial_screening != None:
        
        self.screening = initial_screening[0]
        
        self.quantile = initial_screening[1]
        
        self.df = self.feature_space_screening(self.df)
        
        self.df.columns = self.df.columns.str.replace('-', '_')
        


    # Create the feature values tensor
    self.df_feature_values = torch.tensor(self.df.values).to(self.device)
    
    self.columns = self.df.columns.tolist()
    
    self.variables_indexing = torch.arange(0,len(self.columns)).reshape(1,-1).to(self.device)
    
    self.variables_dict = dict(zip(self.columns, self.variables_indexing.tolist()))

    #Create a dataframe for appending new datavalues
    self.new_features_values = pd.DataFrame()

    #Creating empty tensor and list for single operators (Unary operators)
    self.feature_values_unary = torch.empty(self.df.shape[0],0).to(self.device)
    
    self.feature_names_unary = []

    #creating empty tensor and list for combinations (Binary Operators)
    self.feature_values_binary = torch.empty(self.df.shape[0],0).to(self.device)
    
    self.feature_names_binary = []
    
    #Metrics
    self.rmse_metric = metrics[0]
    
    self.r2_metric = metrics[1]
    
    self.metrics = metrics
    '''
    
    self.test_x = test_x
    
    self.test_y = test_y
    
    self.variable_names = test_variables
    '''
    
    self.pareto_points_identified = torch.empty(0,2).to(self.device)
    
    self.all_points_identified = torch.empty(0,2).to(self.device)
    
    self.p_exp = []
    
    self.np_exp=[]
    
    self.operators_final = torch.empty(0,).to(self.device)
    
    self.operators_final = torch.full((self.df.shape[1],), float('nan')).to(self.device)
    
    self.reference_tensor = self.variables_indexing.clone().reshape(-1, 1)
    
    self.disp=disp
    
    self.pareto=pareto
    
    self.updated_pareto_rmse = torch.empty(0,).to(self.device)
    
    self.updated_pareto_r2 = torch.empty(0,).to(self.device)
    
    self.updated_pareto_complexity = torch.empty(0,).to(self.device)
    
    self.updated_pareto_names =[]
    
    self.update_pareto_coeff =torch.empty(0,).to(self.device)
    
    self.update_pareto_intercepts=torch.empty(0,).to(self.device)
    
    self.dimension = dimension
    
    self.sis_features = sis_features
    
    self.feature_names = feature_names

  '''
  ###############################################################################################################

  Construct all the features that can be constructed using the single operators like log, exp, sqrt etc..

  ###############################################################################################################
  '''
  def feature_space_screening(self,df_sub):

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
        
        return df_screening1
    
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
  
  '''
  #####################################################################
  
  Single variable expansions..
  
  #####################################################################
  '''
  def single_variable(self,operators_set,i):
      
    self.feature_values_unary = torch.empty(self.df.shape[0],0).to(self.device)
    
    self.feature_names_unary = []

    #Looping over operators set to get the new features/predictor variables

    for op in operators_set:

        self.feature_values_11 = torch.empty(self.df.shape[0],0).to(self.device)
        
        feature_names_12 =[]
        
        feature_values_reference = torch.empty(0,).to(self.device)
        
        operators_reference = torch.empty(0,).to(self.device)

        # Performs the exponential transformation of the given feature space
        if op == 'exp':
        
            exp = torch.exp(self.df_feature_values)
            
            self.feature_values_11 = torch.cat((self.feature_values_11,exp),dim=1)
            
            feature_names_12.extend(list(map(lambda x: '(exp('+ x + "))", self.columns)))
            
            
            if i == 1: 
                new_ref = self.reference_tensor[:len(self.columns),:]
                
                operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
            
            else:
                
                new_ref = self.reference_tensor[:self.df_feature_values.shape[1],:]
                
                if self.operators_final.dim()==1:
                    
                    self.operators_final = self.operators_final.unsqueeze(1)
                    
                operators_reference = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1],:].clone()
                
            
                initial_duplicates = self.operators_final[:self.df.shape[1],:].clone()
                
                operators_reference[:,-1]= self.operators_dict[op]
                
                initial_duplicates[:,-1]= self.operators_dict[op]
                
                operators_reference = torch.cat((initial_duplicates,operators_reference),dim=0)
    
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)

        # Performs the natural lograithmic transformation of the given feature space
        elif op =='ln':
            
            ln = torch.log(self.df_feature_values)
            
            self.feature_values_11 = torch.cat((self.feature_values_11,ln),dim=1)
            
            feature_names_12.extend(list(map(lambda x: '(ln('+x + "))", self.columns)))
            
            if i == 1: 
                new_ref = self.reference_tensor[:len(self.columns),:]
                
                operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
            
            else:
                
                new_ref = self.reference_tensor[:self.df_feature_values.shape[1],:]
                
                if self.operators_final.dim()==1:
                
                    self.operators_final = self.operators_final.unsqueeze(1)
                
                operators_reference = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1],:].clone()
                
            
                initial_duplicates = self.operators_final[:self.df.shape[1],:].clone()
                
                operators_reference[:,-1]= self.operators_dict[op]
                
                initial_duplicates[:,-1]= self.operators_dict[op]
                
                operators_reference = torch.cat((initial_duplicates,operators_reference),dim=0)
            
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
            


        # Performs the lograithmic transformation of the given feature space
        elif op =='log':
        
            log10 = torch.log10(self.df_feature_values)
            
            self.feature_values_11 = torch.cat((self.feature_values_11,log10),dim=1)
            
            feature_names_12.extend(list(map(lambda x: '(log('+x + "))", self.columns)))
            
            if i == 1: 
                new_ref = self.reference_tensor[:len(self.columns),:]
                
                operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
            
            else:
                
                new_ref = self.reference_tensor[:self.df_feature_values.shape[1],:]
                
                if self.operators_final.dim()==1:
                    self.operators_final = self.operators_final.unsqueeze(1)
                
                operators_reference = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1],:].clone()
                
            
                initial_duplicates = self.operators_final[:self.df.shape[1],:].clone()
                
                operators_reference[:,-1]= self.operators_dict[op]
                
                initial_duplicates[:,-1]= self.operators_dict[op]
                
                operators_reference = torch.cat((initial_duplicates,operators_reference),dim=0)
            
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
            


        # Performs the power transformations of the feature variables..
        
        elif "pow" in op:
            
            import re
            
            pattern = r'\(([^)]*)\)'
            matches = re.findall(pattern, op)
            op = eval(matches[0])
            
            transformation = torch.pow(self.df_feature_values,op)
            
            self.feature_values_11 = torch.cat((self.feature_values_11,transformation),dim=1)
            
            feature_names_12.extend(list(map(lambda x: '('+x + f")**{matches[0]}", self.columns)))
            
            op = "pow(" + str(Fraction(op)) + ")"
            
            if i == 1: 
                new_ref = self.reference_tensor[:len(self.columns),:]
                operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
            
            else:
                
                new_ref = self.reference_tensor[:self.df_feature_values.shape[1],:]
                if self.operators_final.dim()==1:
                    self.operators_final = self.operators_final.unsqueeze(1)
                operators_reference = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1],:].clone()
                
            
                initial_duplicates = self.operators_final[:self.df.shape[1],:].clone()
                operators_reference[:,-1]= self.operators_dict[op]
                initial_duplicates[:,-1]= self.operators_dict[op]
                operators_reference = torch.cat((initial_duplicates,operators_reference),dim=0)
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)


        # Performs the sine function transformation of the given feature space
        elif op =='sin':
        
            sin = torch.sin(self.df_feature_values)
            
            self.feature_values_11 = torch.cat((self.feature_values_11,sin),dim=1)
            
            feature_names_12.extend(list(map(lambda x: '(sin('+x + "))", self.columns)))
            
            if i == 1: 
                new_ref = self.reference_tensor[:len(self.columns),:]
                operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
            
            else:
                
                new_ref = self.reference_tensor[:self.df_feature_values.shape[1],:]
                if self.operators_final.dim()==1:
                    self.operators_final = self.operators_final.unsqueeze(1)
                operators_reference = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1],:].clone()
                
            
                initial_duplicates = self.operators_final[:self.df.shape[1],:].clone()
                operators_reference[:,-1]= self.operators_dict[op]
                initial_duplicates[:,-1]= self.operators_dict[op]
                operators_reference = torch.cat((initial_duplicates,operators_reference),dim=0)
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
            


        # Performs the hyperbolic sine function transformation of the given feature space
        elif op =='sinh':
        
            sin = torch.sinh(self.df_feature_values)
            
            self.feature_values_11 = torch.cat((self.feature_values_11,sin),dim=1)
            
            feature_names_12.extend(list(map(lambda x: '(sinh('+x + "))", self.columns)))
            
            if i == 1: 
                new_ref = self.reference_tensor[:len(self.columns),:]
                operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
            
            else:
                
                new_ref = self.reference_tensor[:self.df_feature_values.shape[1],:]
                if self.operators_final.dim()==1:
                    self.operators_final = self.operators_final.unsqueeze(1)
                operators_reference = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1],:].clone()
                
            
                initial_duplicates = self.operators_final[:self.df.shape[1],:].clone()
                operators_reference[:,-1]= self.operators_dict[op]
                initial_duplicates[:,-1]= self.operators_dict[op]
                operators_reference = torch.cat((initial_duplicates,operators_reference),dim=0)
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
            


        # Performs the cosine transformation of the given feature space
        elif op =='cos':
        
            cos = torch.cos(self.df_feature_values)
            
            self.feature_values_11 = torch.cat((self.feature_values_11,cos),dim=1)
            
            feature_names_12.extend(list(map(lambda x: '(cos('+x + "))", self.columns)))
            
            if i == 1: 
                new_ref = self.reference_tensor[:len(self.columns),:]
                operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
            
            else:
                
                new_ref = self.reference_tensor[:self.df_feature_values.shape[1],:]
                if self.operators_final.dim()==1:
                    self.operators_final = self.operators_final.unsqueeze(1)
                operators_reference = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1],:].clone()
                
            
                initial_duplicates = self.operators_final[:self.df.shape[1],:].clone()
                operators_reference[:,-1]= self.operators_dict[op]
                initial_duplicates[:,-1]= self.operators_dict[op]
                operators_reference = torch.cat((initial_duplicates,operators_reference),dim=0)
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
            


        # Performs the hyperbolic cosine transformation of the given feature space
        elif op =='cosh':
        
            cos = torch.cosh(self.df_feature_values)
            
            self.feature_values_11 = torch.cat((self.feature_values_11,cos),dim=1)
            
            feature_names_12.extend(list(map(lambda x: '(cosh('+x + "))", self.columns)))
            
            if i == 1: 
                new_ref = self.reference_tensor[:len(self.columns),:]
                operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
            
            else:
                
                new_ref = self.reference_tensor[:self.df_feature_values.shape[1],:]
                if self.operators_final.dim()==1:
                    self.operators_final = self.operators_final.unsqueeze(1)
                operators_reference = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1],:].clone()
                
            
                initial_duplicates = self.operators_final[:self.df.shape[1],:].clone()
                operators_reference[:,-1]= self.operators_dict[op]
                initial_duplicates[:,-1]= self.operators_dict[op]
                operators_reference = torch.cat((initial_duplicates,operators_reference),dim=0)
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
            


        # Performs the hyperbolic tan transformation of the given feature space
        elif op =='tanh':
        
            tanh = torch.tanh(self.df_feature_values)
            
            self.feature_values_11 = torch.cat((self.feature_values_11,tanh),dim=1)
            feature_names_12.extend(list(map(lambda x: '(tanh('+x + "))", self.columns)))
            
            if i == 1: 
                new_ref = self.reference_tensor[:len(self.columns),:]
                operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
            
            else:
                
                new_ref = self.reference_tensor[:self.df_feature_values.shape[1],:]
                if self.operators_final.dim()==1:
                    self.operators_final = self.operators_final.unsqueeze(1)
                operators_reference = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1],:].clone()
                
            
                initial_duplicates = self.operators_final[:self.df.shape[1],:].clone()
                operators_reference[:,-1]= self.operators_dict[op]
                initial_duplicates[:,-1]= self.operators_dict[op]
                operators_reference = torch.cat((initial_duplicates,operators_reference),dim=0)
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
            

            
        # Performs the Inverse transformation of the given feature space
        elif op =='^-1':
        
            reciprocal = torch.reciprocal(self.df_feature_values)
            
            self.feature_values_11 = torch.cat((self.feature_values_11,reciprocal),dim=1)
            
            feature_names_12.extend(list(map(lambda x: '(('+x + ")**-1)", self.columns)))
            if i == 1: 
                new_ref = self.reference_tensor[:len(self.columns),:]
                operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
            
            else:
                
                new_ref = self.reference_tensor[:self.df_feature_values.shape[1],:]
                if self.operators_final.dim()==1:
                    self.operators_final = self.operators_final.unsqueeze(1)
                operators_reference = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1],:].clone()
                
            
                initial_duplicates = self.operators_final[:self.df.shape[1],:].clone()
                operators_reference[:,-1]= self.operators_dict[op]
                initial_duplicates[:,-1]= self.operators_dict[op]
                operators_reference = torch.cat((initial_duplicates,operators_reference),dim=0)
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
            


        # Performs the Inverse exponential transformation of the given feature space
        elif op =='exp(-1)':
        
            exp = torch.exp(self.df_feature_values)
            
            expreciprocal = torch.reciprocal(exp)
            
            self.feature_values_11 = torch.cat((self.feature_values_11,expreciprocal),dim=1)
            
            feature_names_12.extend(list(map(lambda x: '(exp(-'+x + "))", self.columns)))
            
            if i == 1: 
                new_ref = self.reference_tensor[:len(self.columns),:]
                operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
            
            else:
                
                new_ref = self.reference_tensor[:self.df_feature_values.shape[1],:]
                if self.operators_final.dim()==1:
                    self.operators_final = self.operators_final.unsqueeze(1)
                operators_reference = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1],:].clone()
                
            
                initial_duplicates = self.operators_final[:self.df.shape[1],:].clone()
                operators_reference[:,-1]= self.operators_dict[op]
                initial_duplicates[:,-1]= self.operators_dict[op]
                operators_reference = torch.cat((initial_duplicates,operators_reference),dim=0)
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
            

            
        elif op =='+1':
            add1 = self.df_feature_values + 1
            self.feature_values_11 = torch.cat((self.feature_values_11,add1),dim=1)
            feature_names_12.extend(list(map(lambda x: '('+x + "+1)", self.columns)))
            
            if i == 1: 
                new_ref = self.reference_tensor[:len(self.columns),:]
                operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
            
            else:
                
                new_ref = self.reference_tensor[:self.df_feature_values.shape[1],:]
                if self.operators_final.dim()==1:
                    self.operators_final = self.operators_final.unsqueeze(1)
                operators_reference = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1],:].clone()
                
            
                initial_duplicates = self.operators_final[:self.df.shape[1],:].clone()
                operators_reference[:,-1]= self.operators_dict[op]
                initial_duplicates[:,-1]= self.operators_dict[op]
                operators_reference = torch.cat((initial_duplicates,operators_reference),dim=0)
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
            

            
        elif op =='-1':
        
            sub1 = self.df_feature_values - 1
            
            self.feature_values_11 = torch.cat((self.feature_values_11,sub1),dim=1)
            
            feature_names_12.extend(list(map(lambda x: '('+x + "-1)", self.columns)))
            
            if i == 1: 
                new_ref = self.reference_tensor[:len(self.columns),:]
                operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
            
            else:
                
                new_ref = self.reference_tensor[:self.df_feature_values.shape[1],:]
                if self.operators_final.dim()==1:
                    self.operators_final = self.operators_final.unsqueeze(1)
                operators_reference = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1],:].clone()
                
            
                initial_duplicates = self.operators_final[:self.df.shape[1],:].clone()
                operators_reference[:,-1]= self.operators_dict[op]
                initial_duplicates[:,-1]= self.operators_dict[op]
                operators_reference = torch.cat((initial_duplicates,operators_reference),dim=0)
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)

        elif op =='/2':
        
            div2 = self.df_feature_values/2
            
            self.feature_values_11 = torch.cat((self.feature_values_11,div2),dim=1)
            
            feature_names_12.extend(list(map(lambda x: '('+x + "/2)", self.columns)))
            
            if i == 1: 
                new_ref = self.reference_tensor[:len(self.columns),:]
                operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
            
            else:
                
                new_ref = self.reference_tensor[:self.df_feature_values.shape[1],:]
                if self.operators_final.dim()==1:
                    self.operators_final = self.operators_final.unsqueeze(1)
                operators_reference = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1],:].clone()
                
            
                initial_duplicates = self.operators_final[:self.df.shape[1],:].clone()
                operators_reference[:,-1]= self.operators_dict[op]
                initial_duplicates[:,-1]= self.operators_dict[op]
                operators_reference = torch.cat((initial_duplicates,operators_reference),dim=0)
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)

        

        self.feature_values_unary = torch.cat((self.feature_values_unary,self.feature_values_11),dim=1)
        self.feature_names_unary.extend(feature_names_12)
        
        
        self.reference_tensor = torch.cat((self.reference_tensor, feature_values_reference), dim=0)
        self.reference_tensor = self.clean_tensor(self.reference_tensor)
        
        if self.operators_final.dim()==1 : self.operators_final = self.operators_final.unsqueeze(1)
        
        if operators_reference.dim()==1: operators_reference = operators_reference.unsqueeze(1)
        
        if self.operators_final.shape[1] == operators_reference.shape[1]:
        
            self.operators_final = torch.cat((self.operators_final, operators_reference))
            
        else:
            
            additional_columns = torch.full((self.operators_final.size(0), abs(operators_reference.shape[1]-self.operators_final.shape[1])), float('nan'))
            
            self.operators_final = torch.cat((self.operators_final,additional_columns),dim=1)
            
            self.operators_final = torch.cat((self.operators_final, operators_reference))
        
        self.operators_final = self.clean_tensor(self.operators_final)
        
        

        del self.feature_values_11, feature_names_12


        
    return self.feature_values_unary, self.feature_names_unary



  '''
  ################################################################################################

  Defining method to perform the combinations of the variables with the initial feature set
  ################################################################################################
  '''
  def combinations(self,operators_set,i):
      
      #creating empty tensor and list for combinations (Binary Operators)
      self.feature_values_binary = torch.empty(self.df.shape[0],0).to(self.device)
      
      self.feature_names_binary = []

      for op in operators_set:
          
          feature_values_reference = torch.empty(0,).to(self.device)
          
          operators_reference = torch.empty(0,).to(self.device)
          
          combinations1 = list(combinations(self.columns,2))

          combinations2 = torch.combinations(torch.arange(self.df_feature_values.shape[1]),2)

          comb_tensor = self.df_feature_values.T[combinations2,:]

          #Reshaping to match
          x_p = comb_tensor.permute(0,2,1)

          
          del comb_tensor
          
          self.feature_values11 = torch.empty(self.df.shape[0],0).to(self.device)
          
          feature_names_11 = []

          # Performs the addition transformation of feature space with the combinations generated
          if op =='+':
          
              sum = torch.sum(x_p,dim=2).T
              
              self.feature_values11 = torch.cat((self.feature_values11,sum),dim=1)
              
              feature_names_11.extend(list(map(lambda comb: '('+'+'.join(comb)+')', combinations1)))
              
              del combinations1

              new_ref = torch.cat([self.reference_tensor[combinations2[:, 0]], 
                                 self.reference_tensor[combinations2[:, 1]]], dim=1)
              
              max_cols = max(self.reference_tensor.shape[1],new_ref.shape[1])
              self.reference_tensor = torch.cat([self.reference_tensor, torch.full((self.reference_tensor.size(0), max_cols - self.reference_tensor.size(1)), float('nan'))], dim=1)

              feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
              if i == 1: 
                  
                  operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                  
              else:
                  
                  threshold = self.df.shape[1]
                  condition = (combinations2[:, 0] < threshold) & (combinations2[:, 1] < threshold)
                  indices = torch.nonzero(condition).squeeze()
                  non_indices = torch.nonzero(~condition).squeeze()
                  try:
                      op1 = torch.full((len(indices),), self.operators_dict[op])
                  except: 
                      op1 = torch.full((1,), self.operators_dict[op])
                  combinations2[non_indices] = torch.where(combinations2[non_indices] >= threshold, combinations2[non_indices] - threshold, combinations2[non_indices])
                  op2 = self.operators_final[combinations2[non_indices]]
                  
                  op2 = self.operators_final[self.df.shape[1]:][combinations2[non_indices]]
                  
                  # Determine the number of columns in tensor2
                  
                  op2 = op2.view(op2.size(0), -1)
                  num_columns_tensor2 = op2.size(1)

                  op1 = op1.unsqueeze(1)  # Convert to shape (N, 1)
                  nan_column = torch.full((op1.size(0), num_columns_tensor2 - 1), float('nan'))
                  op1 = torch.cat((op1, nan_column), dim=1)
                  
                  op3 = torch.empty(len(feature_names_11),op2.shape[1])
                  
                  mask = torch.ones(len(feature_names_11), dtype=torch.bool)
                  mask[indices] = False
                

                  op3[mask] = op2
                      
                  
                  op3[indices] = op1
                  
                 
                  
                  nan_column = torch.full((op3.size(0), 1), float('nan'))
                  op3 = torch.cat((op3,nan_column),dim=1)
                  combinations2 = torch.combinations(torch.arange(self.df_feature_values.shape[1]),2)
                  combinations2[non_indices] = combinations2[non_indices] - self.df.shape[1]
                  negative_indices = torch.nonzero(combinations2 < 0, as_tuple=False)
                  mask = torch.ones(op3.size(0), dtype=torch.bool)
                  mask[negative_indices[:,0]]=False
                  op3[mask,-1] = self.operators_dict[op]
                  op3[indices,-1] = float('nan')
                  #pdb.set_trace()


                  if self.operators_final.dim() ==1:  self.operators_final = self.operators_final.unsqueeze(1)
                  nan_column = torch.full((self.operators_final.size(0), op3.shape[1] - self.operators_final.shape[1]), float('nan'))
                  self.operators_final = torch.cat((self.operators_final, nan_column), dim=1)
                  
                  operators_reference = torch.cat((operators_reference, op3))
                  del combinations2
                  
          # Performs the subtraction transformation of feature space with the combinations generated
          elif op =='-':
          
              sub = torch.sub(x_p[:,:,0],x_p[:,:,1]).T
              
              self.feature_values11 = torch.cat((self.feature_values11,sub),dim=1)
              
              feature_names_11.extend(list(map(lambda comb: '('+'-'.join(comb)+')', combinations1)))
              
              del combinations1
              
              new_ref = torch.cat([self.reference_tensor[combinations2[:, 0]], 
                                 self.reference_tensor[combinations2[:, 1]]], dim=1)
              
              max_cols = max(self.reference_tensor.shape[1],new_ref.shape[1])
              self.reference_tensor = torch.cat([self.reference_tensor, torch.full((self.reference_tensor.size(0), max_cols - self.reference_tensor.size(1)), float('nan'))], dim=1)
              
              feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
              if i == 1: 
                  operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
              else:
                  
                  threshold = self.df.shape[1]
                  condition = (combinations2[:, 0] < threshold) & (combinations2[:, 1] < threshold)
                  #pdb.set_trace()
                  indices = torch.nonzero(condition).squeeze()
                  non_indices = torch.nonzero(~condition).squeeze()
                  try:
                      op1 = torch.full((len(indices),), self.operators_dict[op])
                  except: 
                      op1 = torch.full((1,), self.operators_dict[op])
                  combinations2[non_indices] = torch.where(combinations2[non_indices] >= threshold, combinations2[non_indices] - threshold, combinations2[non_indices])
                  op2 = self.operators_final[combinations2[non_indices]]
                  
                  op2 = self.operators_final[self.df.shape[1]:][combinations2[non_indices]]
                  
                  # Determine the number of columns in tensor2
                  
                  op2 = op2.view(op2.size(0), -1)
                  num_columns_tensor2 = op2.size(1)

                  op1 = op1.unsqueeze(1)  # Convert to shape (N, 1)
                  nan_column = torch.full((op1.size(0), num_columns_tensor2 - 1), float('nan'))
                  op1 = torch.cat((op1, nan_column), dim=1)
                  
                  op3 = torch.empty(len(feature_names_11),op2.shape[1])
                  
                  mask = torch.ones(len(feature_names_11), dtype=torch.bool)
                  mask[indices] = False
                

                  op3[mask] = op2
                      
                  
                  op3[indices] = op1
                  
                 
                  
                  nan_column = torch.full((op3.size(0), 1), float('nan'))
                  op3 = torch.cat((op3,nan_column),dim=1)
                  combinations2 = torch.combinations(torch.arange(self.df_feature_values.shape[1]),2)
                  combinations2[non_indices] = combinations2[non_indices] - self.df.shape[1]
                  negative_indices = torch.nonzero(combinations2 < 0, as_tuple=False)
                  mask = torch.ones(op3.size(0), dtype=torch.bool)
                  mask[negative_indices[:,0]]=False
                  op3[mask,-1] = self.operators_dict[op]
                  op3[indices,-1] = float('nan')
                  #pdb.set_trace()


                  if self.operators_final.dim() ==1:  self.operators_final = self.operators_final.unsqueeze(1)
                  nan_column = torch.full((self.operators_final.size(0), op3.shape[1] - self.operators_final.shape[1]), float('nan'))
                  self.operators_final = torch.cat((self.operators_final, nan_column), dim=1)
                  
                  operators_reference = torch.cat((operators_reference, op3))
                  
                  del combinations2

          # Performs the division transformation of feature space with the combinations generated
          elif op == '/':
          
              div1 = torch.div(x_p[:,:,0],x_p[:,:,1]).T
              
              div2 = torch.div(x_p[:,:,1],x_p[:,:,0]).T
              
              self.feature_values11 = torch.cat((self.feature_values11,div1,div2),dim=1)
              
              feature_names_11.extend(list(map(lambda comb: '('+'/'.join(comb)+')', combinations1)))
              
              feature_names_11.extend(list(map(lambda comb: '('+'/'.join(comb[::-1])+')', combinations1)))
              
              del combinations1
              
              new_ref = torch.cat([self.reference_tensor[combinations2[:, 0]], 
                                 self.reference_tensor[combinations2[:, 1]]], dim=1)
              
              max_cols = max(self.reference_tensor.shape[1],new_ref.shape[1])
              
              self.reference_tensor = torch.cat([self.reference_tensor, torch.full((self.reference_tensor.size(0), max_cols - self.reference_tensor.size(1)), float('nan'))], dim=1)

              feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
              
              feature_values_reference = feature_values_reference.repeat(2,1)
              
              if i == 1: 
                  operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                  
                  operators_reference = torch.cat((operators_reference,operators_reference))
                  
              else:
                  
                  threshold = self.df.shape[1]
                  
                  condition = (combinations2[:, 0] < threshold) & (combinations2[:, 1] < threshold)
                  
                  indices = torch.nonzero(condition).squeeze()
                  
                  non_indices = torch.nonzero(~condition).squeeze()
                  
                  try:
                  
                      op1 = torch.full((len(indices),), self.operators_dict[op])
                  except:
                      op1 = torch.full((1,), self.operators_dict[op])
                  
                  combinations2[non_indices] = torch.where(combinations2[non_indices] >= threshold, combinations2[non_indices] - threshold, combinations2[non_indices])
                  
                  op2 = self.operators_final[combinations2[non_indices]]
                  
                  op2 = self.operators_final[self.df.shape[1]:][combinations2[non_indices]]
                  
                  op2 = op2.view(op2.size(0), -1)
                  #pdb.set_trace()
                  try:
                      indices = torch.cat((indices,indices+ op2.shape[0]))
                  except:
                      s = torch.tensor([indices])
                      indices = torch.cat((s,s+ op2.shape[0]))
                  
                  op2 = op2.repeat(2,1)
                  
                  num_columns_tensor2 = op2.size(1)

                  op1 = op1.unsqueeze(1) 
                  
                  nan_column = torch.full((op1.size(0), num_columns_tensor2 - 1), float('nan'))
                  
                  op1 = torch.cat((op1, nan_column), dim=1)
                  
                  op1 = op1.repeat(2,1)
                  
                  op3 = torch.empty(len(feature_names_11),op2.shape[1])
                  
                  mask = torch.ones(len(feature_names_11), dtype=torch.bool)
                  
                  mask[indices] = False

                  op3[mask] = op2

                  op3[indices] = op1

                  nan_column = torch.full((op3.size(0), 1), float('nan'))
                  
                  op3 = torch.cat((op3,nan_column),dim=1)
                  
                  combinations2 = torch.combinations(torch.arange(self.df_feature_values.shape[1]),2)
                  
                  combinations2[non_indices] = combinations2[non_indices] - self.df.shape[1]
                  
                  negative_indices = torch.nonzero(combinations2 < 0, as_tuple=False)
                  
                  mask = torch.ones(op3.size(0), dtype=torch.bool)
                  
                  mask[negative_indices[:,0]]=False
                  
                  op3[mask,-1] = self.operators_dict[op]
                  
                  op3[indices,-1] = float('nan')
                  
                  if self.operators_final.dim() ==1:  self.operators_final = self.operators_final.unsqueeze(1)
                  
                  nan_column = torch.full((self.operators_final.size(0), op3.shape[1] - self.operators_final.shape[1]), float('nan'))
                  
                  self.operators_final = torch.cat((self.operators_final, nan_column), dim=1)
                  
                  operators_reference = torch.cat((operators_reference, op3))
                  
                  del combinations2
                  

          # Performs the multiplication transformation of feature space with the combinations generated
          elif op == '*':
          
              mul = torch.multiply(x_p[:,:,0],x_p[:,:,1]).T
              
              self.feature_values11 = torch.cat((self.feature_values11,mul),dim=1)
              
              feature_names_11.extend(list(map(lambda comb: '('+'*'.join(comb)+')', combinations1)))
              
              del combinations1
              
              new_ref = torch.cat([self.reference_tensor[combinations2[:, 0]],self.reference_tensor[combinations2[:, 1]]], dim=1)
              
              max_cols = max(self.reference_tensor.shape[1],new_ref.shape[1])
              
              self.reference_tensor = torch.cat([self.reference_tensor, torch.full((self.reference_tensor.size(0), max_cols - self.reference_tensor.size(1)), float('nan'))], dim=1)
              
              feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
              
              if i == 1: 
                  
                  operators_reference = torch.cat((operators_reference, torch.full((new_ref.shape[0],), self.operators_dict[op])))
                  
              else:
                  
                  threshold = self.df.shape[1]
                  
                  condition = (combinations2[:, 0] < threshold) & (combinations2[:, 1] < threshold)
                  
                  indices = torch.nonzero(condition).squeeze()
                  
                  non_indices = torch.nonzero(~condition).squeeze()
                  
                  op1 = torch.full((len(indices),), self.operators_dict[op])
                  
                  combinations2[non_indices] = torch.where(combinations2[non_indices] >= threshold, combinations2[non_indices] - threshold, combinations2[non_indices])
                  
                  op2 = self.operators_final[combinations2[non_indices]]
                  
                  op2 = self.operators_final[self.df.shape[1]:][combinations2[non_indices]]
                  
                  op2 = op2.view(op2.size(0), -1)
                  
                  num_columns_tensor2 = op2.size(1)

                  op1 = op1.unsqueeze(1)  
                  
                  nan_column = torch.full((op1.size(0), num_columns_tensor2 - 1), float('nan'))
                  
                  op1 = torch.cat((op1, nan_column), dim=1)
                  
                  op3 = torch.empty(len(feature_names_11),op2.shape[1])
                  
                  mask = torch.ones(len(feature_names_11), dtype=torch.bool)
                  
                  mask[indices] = False

                  op3[mask] = op2

                  op3[indices] = op1

                  nan_column = torch.full((op3.size(0), 1), float('nan'))
                  
                  op3 = torch.cat((op3,nan_column),dim=1)
                  
                  combinations2 = torch.combinations(torch.arange(self.df_feature_values.shape[1]),2)
                  
                  combinations2[non_indices] = combinations2[non_indices] - self.df.shape[1]
                  
                  negative_indices = torch.nonzero(combinations2 < 0, as_tuple=False)
                  
                  mask = torch.ones(op3.size(0), dtype=torch.bool)
                  
                  mask[negative_indices[:,0]]=False
                  
                  op3[mask,-1] = self.operators_dict[op]
                  
                  op3[indices,-1] = float('nan')

                  if self.operators_final.dim() ==1:  self.operators_final = self.operators_final.unsqueeze(1)
                  
                  nan_column = torch.full((self.operators_final.size(0), op3.shape[1] - self.operators_final.shape[1]), float('nan'))
                  
                  self.operators_final = torch.cat((self.operators_final, nan_column), dim=1)
                  
                  operators_reference = torch.cat((operators_reference, op3))
                  
                  del combinations2

          self.feature_values_binary = torch.cat((self.feature_values_binary,self.feature_values11),dim=1)
          
          self.feature_names_binary.extend(feature_names_11)
          
          self.reference_tensor = torch.cat((self.reference_tensor, feature_values_reference), dim=0)
          
          self.reference_tensor = self.clean_tensor(self.reference_tensor)
          
          if self.operators_final.dim()==1 : self.operators_final = self.operators_final.unsqueeze(1)
          
          if operators_reference.dim()==1: operators_reference = operators_reference.unsqueeze(1)
          
          if self.operators_final.shape[1] == operators_reference.shape[1]:
          
              self.operators_final = torch.cat((self.operators_final, operators_reference))
              
          else:
              
              additional_columns = torch.full((self.operators_final.size(0), abs(operators_reference.shape[1]-self.operators_final.shape[1])), float('nan'))
              
              self.operators_final = torch.cat((self.operators_final,additional_columns),dim=1)
              
              self.operators_final = torch.cat((self.operators_final, operators_reference))
              
          
          self.operators_final = self.clean_tensor(self.operators_final)
          
          
          del self.feature_values11,feature_names_11
          
          
      return self.feature_values_binary,self.feature_names_binary #created_space


  '''
  ##########################################################################################################

  Creating the space based on the given set of conditions

  ##########################################################################################################

  '''

  def feature_space(self):


    # Split the operator set into combinations set and unary set
    basic_operators = [op for op in self.operators if op in ['+', '-', '*', '/']]
    
    other_operators = [op for op in self.operators if op not in ['+', '-', '*', '/']]
    
    
    if self.no_of_operators == None:
        
        #if self.disp: print('############################################################# Implementing Automatic Expansion and construction of sparse models..!!! ######################################################################')
        
        from .Regressor import Regressor
        
        i = 1
        
        start_time = time.time()
        
        values, names = self.combinations(basic_operators,i)
    
        # Performs the feature space expansion based on the unary operator set provided
        values1, names1 = self.single_variable(other_operators,i)
    
        features_created = torch.cat((values,values1),dim=1)
        
        del values, values1
        
        names2 = names + names1
        
        del names,names1
        
        self.df_feature_values = torch.cat((self.df_feature_values,features_created),dim=1)
        
        self.columns.extend(names2)
        
        del features_created,names2
        
        if self.disp:
        
            print('****************************** Initial Feature Expansion Completed with feature space size: ',self.df_feature_values.shape[1],'*********************************************** \n')
            
            print('**************************************** Time taken to create the space is:::', time.time()-start_time, ' Seconds ********************************************\n')
            
        
        #self.dimension=None
        
        #self.sis_features=None
        
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
        
        
        rmse1, equation1,r21,r,c,n,intercepts,coeffs,r2_value =  Regressor(self.df_feature_values,self.Target_column,self.columns,complexity,self.dimension,self.sis_features,self.device,metrics = self.metrics).regressor_fit()
        
        additional_columns = torch.full((1, abs(coeffs.shape[1])), float('nan'))
        
        additional_columns[:,0] = 1
        
        coeffs = torch.cat((additional_columns,coeffs))
        
        intercepts= torch.cat((torch.tensor([0]),intercepts))
        
        s= pareto(r,c,final_pareto='no').pareto_front()
        
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
            
            #print('Equation:',equation1)
            
            return rmse1,equation1,r21,df_sorted 
        
        i = 2
        
        while True:
            
            values, names = self.combinations(basic_operators,i)

        
            # Performs the feature space expansion based on the unary operator set provided
            values1, names1 = self.single_variable(other_operators,i)

        
            features_created = torch.cat((values,values1),dim=1)
            
            del values, values1
            
            names2 = names + names1
            
            del names,names1
            
            self.df_feature_values = torch.cat((self.df_feature_values,features_created),dim=1)
            
            self.columns.extend(names2)
            
            del features_created,names2
            
            if self.disp:
            
                print(f'************************************ {i} Feature Expansion Completed with feature space size:::',self.df_feature_values.shape[1],' *********************************************************** \n')
                
                print('****************************************** Time taken to create the space is:::', time.time()-start_time, ' Seconds *************************************************** \n')
                
            if self.df_feature_values.shape[1] <10000:
            
                
                unique_columns, indices = torch.unique(self.df_feature_values, sorted=False,dim=1, return_inverse=True)
                
                # Get the indices of the unique columns
                unique_indices = indices.unique()
      
                # Remove duplicate columns
                self.df_feature_values = self.df_feature_values[:, unique_indices]
                
                
                # Remove the corresponding elements from the list of feature names..
                self.columns = [self.columns[i] for i in unique_indices.tolist()]
                
                
                self.reference_tensor = self.reference_tensor[unique_indices,:]
                
                if self.operators_final.dim() ==1 : self.operators_final = self.operators_final[unique_indices] 
                
                else: self.operators_final = self.operators_final[unique_indices,:] 
            
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
            
            
            rmse, equation,r2,r,c,n,intercepts,coeffs,r2_value =  Regressor(self.df_feature_values,self.Target_column,self.columns,complexity,self.dimension,self.sis_features,self.device,metrics = self.metrics).regressor_fit()
            
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
            if i >=2 and self.df_feature_values.shape[1]>2000:
                
                print('Expanded feature space is::',self.df_feature_values.shape[1])
                
                
                print('!!Warning:: Further feature expansions result in memory consumption, Please provide the input to consider feature expansion or to exit the run with the sparse models created!!!')
                
                response = input("Do you wish to continue (yes/no)? ").strip().lower()
                
                if response == 'no' or response == 'n': 
                    
                    print("Exiting based on user input.")
                    
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
        
        #print('Equation:',equation)
        
        return rmse,equation,r2,df_sorted

    else:
        
        for i in range(1,self.no_of_operators):
            
            start_time = time.time()
            
            if self.disp: print(f'*********************************   Starting {i} level of feature expansion******************************************** \n')
    
            #Performs the feature space expansion based on the binary operator set provided
            values, names = self.combinations(basic_operators,i)
        
            # Performs the feature space expansion based on the unary operator set provided
            values1, names1 = self.single_variable(other_operators,i)
            
        
            features_created = torch.cat((values,values1),dim=1)
            
            del values, values1
            
            names2 = names + names1
            
            del names,names1
            
            self.df_feature_values = torch.cat((self.df_feature_values,features_created),dim=1)
            
            self.columns.extend(names2)
            
            del features_created,names2
            
            unique_columns, indices = torch.unique(self.df_feature_values, sorted=False,dim=1, return_inverse=True)
            
            # Get the indices of the unique columns
            unique_indices = indices.unique()
      
            # Remove duplicate columns
            self.df_feature_values = self.df_feature_values[:, unique_indices]
            
            
            # Remove the corresponding elements from the list of feature names..
            self.columns = [self.columns[i] for i in unique_indices.tolist()]
            
            self.reference_tensor = self.reference_tensor[unique_indices,:]
            
            if self.operators_final.dim() ==1 : self.operators_final = self.operators_final[unique_indices] 
            else: self.operators_final = self.operators_final[unique_indices,:] 
            
            if self.disp:
                print(f'**************************** {i} Feature Expansion Completed with feature space size:::',self.df_feature_values.shape[1],'************************************************* \n')
                
                if self.feature_names: 
                    
                    print('Feature Names:', self.columns)
                    
                    print('\n \n \n')
                print('****************************************** Time taken to create the space is:::', time.time()-start_time, ' Seconds********************************************* \n')
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
                
            
        return self.df_feature_values, self.Target_column,self.columns,complexity
    
