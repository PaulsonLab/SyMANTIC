#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:59:07 2024

@author: muthyala.7
"""

'''
##############################################################################################

Importing the required libraries 

##############################################################################################
'''

import torch

import warnings

warnings.filterwarnings('ignore')

import itertools

import time 

import torch.nn as nn

import torch.optim as optim

import pdb

import sympy as sp

import numpy as np


from .pareto_new import pareto


class Regressor:
    
    def __init__(self,x,y,names,dimensionality,complexity,output_dim = None,dimension=None,sis_features=10,device='cpu',metrics =[0.06,0.995],disp=False,quantiles = None):

        '''
        ###################################################################################################################

        x, y, names - are the outputs of the Feature Expansion class which defines the expanded feature space, target tensor, names of the expanded features to use in the equation

        dimension - defines the number of terms in the linear equation generation 

        sis_features - defines the number of top features needs to be considered for building the equation

        ###################################################################################################################
        '''
        self.device = device
        
        self.x = x.to(self.device)
        
        self.y = y.to(self.device)
        
        self.complexity = complexity
        
        self.dimensionality = dimensionality
        
        self.output_dim = output_dim
        
        self.names = names
        
        if self.output_dim!=None:
            
            self.get_dimensions_list()
            
            self.x = self.x[:,self.dimension_less]
            
            x = pd.Series(self.names)
            
            self.names = x.iloc[self.dimension_less].tolist()
        
        if dimension !=None: 
            
            self.dimension = dimension
            
            self.sis_features = sis_features
            
        else: 
            self.dimension = 3 #Maximum terms we will be looking at...
            
            self.sis_features = 10
        

        # Transform the features into standardized format
        self.x_mean = self.x.mean(dim=0)
        
        self.x_std = self.x.std(dim=0)
        
        self.y_mean = self.y.mean()

        # Transform the target variable value to mean centered
        self.y_centered = self.y - self.y_mean
        
        self.x_standardized = ((self.x - self.x_mean)/self.x_std)

        self.scores = []
        
        self.indices = torch.arange(1, (self.dimension*self.sis_features+1)).view(self.dimension*self.sis_features,1).to(self.device)
        
        self.residual = torch.empty(self.y_centered.shape).to(self.device)
        
        self.x_std_clone = torch.clone(self.x_standardized)
        
        self.rmse_metric = metrics[0]
        
        self.r2_metric = metrics[1]
        '''
        
        self.test_x = test_x
        
        self.test_y = test_y
        '''
        
        self.earlier_pareto_rmse = torch.empty(0,)
        
        self.earlier_pareto_r2 = torch.empty(0,)
        
        self.earlier_pareto_complexity = torch.empty(0,)
        
        self.pareto_names =[]
        
        self.pareto_coeffs = torch.empty(0,)
        
        self.pareto_intercepts = torch.empty(0,)
        
        if self.x.shape[1]>1000: self.sis_features1 = 1000
        
        else: self.sis_features1 = self.x.shape[1]
        
        self.disp = disp
        
        
        if quantiles !=None : self.quantiles = quantiles
        
        else: self.quantiles  = [0.10, 0.20, 0.3,0.40,0.50,0.60,0.70,0.80,0.90,1.0]

    

    def get_dimensions_list(self):
            
            #get the same dimensions from the list along with their index position.. 
            result ={}
            
            for index, value in enumerate(self.dimensionality):
                
                if value not in result:
                    
                    result[value] = []
                    
                result[value].append(index)
                
            
            
            if self.output_dim in result.keys():
                
                
                #if self.disp: print('************************************************ Extraction of target dimension feature variables found.., performing the regression!!.. ************************************************ \n')
                
                self.dimension_less = result[self.output_dim]
                
                del result[self.output_dim]
                
                if self.disp: print(f'************************************************ {len(self.dimension_less)} output dimension feature variables found in the given list!! ************************************************ \n')
            
                self.dimensions_index_dict = result
                
                del result
                
                
                return self.dimensions_index_dict, self.dimension_less
            
            else:
                
                if self.disp: print('No target dimension feature variables found.. exiting the program..')
                sys.exit()

    '''
    #######################################################################################################

    Constructs the linear equation based on the number of top sis features and the dimension requested.

    #######################################################################################################
    '''
    def higher_dimension(self,iteration):

        #Indices values that needs to be assinged zero 
        ind = (self.indices[:,-1][~torch.isnan(self.indices[:,-1])]).to(self.device)

        self.x_standardized[:,ind.tolist()] = 0

        scores= torch.abs(torch.mm(self.residual,self.x_standardized))

        scores[torch.isnan(scores)] = 0

        self.x_standardized[:,ind.tolist()] = self.x_std_clone[:,ind.tolist()]
        
        
        quantile_values = torch.quantile(self.complexity, torch.tensor(self.quantiles))
        
        '''
        
        try:
            
            quantile_values = torch.quantile(self.complexity, torch.tensor(quantiles))
            
        except:
            
            print('********************* Changing to manual partitions because of large tensor size which hinders at quantile calculation **************************************** \n')
        
            bins, digitized = (lambda t, n: (b := torch.arange(t.min(), t.max() + (w := (t.max() - t.min()) / n), w).add_(1e-6), torch.bucketize(t, b)))(self.complexity, len(quantiles)-1)

            # The upper edges of the bins correspond to the quantiles
            quantile_values = bins[1:]
        '''
        
        earlier_pareto_rmse = torch.empty(0,)
        
        earlier_pareto_complexity = torch.empty(0,)
        
        for i in range(len(quantile_values)):
            
            s = time.time()
            
            self.indices = self.indices_clone
            
            if i == 0 : 
                
                ind = torch.where(self.complexity <= quantile_values[i])[0]
                
                scores1 = scores[:,ind]
                
            else: 
                
                ind = torch.where((self.complexity > quantile_values[i-1])&(self.complexity <= quantile_values[i]))[0]
                
                scores1 = scores[:,ind]
                
            if self.quantiles[i] == 1.0: 
                
                ind = torch.where(self.complexity <= quantile_values[i])[0]
                
                scores1 = scores

                
            comp1 = self.complexity#[ind]
            
            if scores1.size()[1]==0:
                
                continue
            
            try:
                
                sorted_scores, sorted_indices = torch.topk(scores1,k=self.sis_features)
                
            except:
                
                sorted_scores, sorted_indices = torch.topk(scores1,k=len(scores1))
                            
            

            sorted_indices = sorted_indices.T
            
            sorted_indices_earlier = self.indices[:((iteration-1)*self.sis_features),(iteration-1)].unsqueeze(1)
    
            sorted_indices = torch.cat((sorted_indices_earlier,sorted_indices),dim=0)
    
            if sorted_indices.shape[0] < self.indices.shape[0]:
                
                remaining = (self.sis_features*self.dimension) - int(sorted_indices.shape[0])
                
                nan = torch.full((remaining,1),float('nan')).to(self.device)
                
                sorted_indices = torch.cat((sorted_indices,nan),dim=0)
                
                self.indices = torch.cat((self.indices,sorted_indices),dim=1)
    
            else:
                
                self.indices = torch.cat((self.indices,sorted_indices),dim=1)
    
            comb1 = self.indices[:,-1][~torch.isnan(self.indices[:,-1])]
            
            combinations_generated = torch.combinations(comb1,(int(self.indices.shape[1])-1))
            
            
            y_centered_clone = self.y_centered.unsqueeze(1).repeat(len(combinations_generated.tolist()),1,1).to(self.device)
            
            comb_tensor = self.x_standardized.T[combinations_generated.tolist(),:]
            
            x_p = comb_tensor.permute(0,2,1)
            
            comp2 = comp1[combinations_generated.to(torch.int)]
            
            comp2 = torch.sum(comp2,dim=1)
            
            comp2 = comp2+i
            
            has_nan_inf = torch.logical_or(
                torch.isnan(x_p).any(dim=1, keepdim=True).any(dim=2, keepdim=True),
                torch.isinf(x_p).any(dim=1, keepdim=True).any(dim=2, keepdim=True)
                )
            
            x_p = torch.where(has_nan_inf,torch.zeros_like(x_p),x_p)
            
            try:
                
                sol,_,_,_ = torch.linalg.lstsq(x_p,y_centered_clone)
                
            except:
                
                x2_inv = torch.linalg.pinv(x_p)
                
                sol = x2_inv@y_centered_clone
                
                sol[torch.isnan(sol)] = 0
            

            predicted = torch.matmul(x_p,sol)
            
            residuals = y_centered_clone - predicted
            
            square = torch.square(residuals)
            
            mean = torch.mean(square,dim=1,keepdim=True)
            
            features_rmse = torch.sqrt(mean)[:,0,0]
            
            features_r2 = 1 - (torch.sum(torch.square(residuals),dim=1)/torch.sum(torch.square(self.y_centered)))
            
            s= pareto(features_rmse,comp2).pareto_front()
            
            coeff = torch.squeeze(sol).unsqueeze(1)
            
            coeff = coeff.squeeze(1)
            
            coeff1 = coeff.clone()
            
            combinations = combinations_generated.long()
            
            std = self.x_std[combinations]
            
            coeff = coeff/std
            
            xx = self.x_mean[combinations_generated.to(torch.int)]
            yy = self.x_std[combinations_generated.to(torch.int)]
            
            nn = xx/yy
            
            ss1 = nn*coeff1
            
            ss2 = torch.sum(ss1,dim=1)
            
            non_std_intercepts = self.y.mean().repeat(coeff1.shape[0]) -  ss2
            
            
            self.earlier_pareto_rmse = torch.cat((self.earlier_pareto_rmse,features_rmse[s]),dim=0)
            
            self.earlier_pareto_r2 = torch.cat((self.earlier_pareto_r2,features_r2[s].flatten()),dim=0)
            
            self.earlier_pareto_complexity = torch.cat((self.earlier_pareto_complexity,comp2[s]))
            
            if coeff.shape[1] == self.pareto_coeffs.shape[1]:
                
                self.pareto_coeffs = torch.cat((self.pareto_coeffs,coeff[s]))
            else:
                additional_columns = torch.full((self.pareto_coeffs.size(0), abs(coeff.shape[1]-self.pareto_coeffs.shape[1])), float('nan'))
                
                self.pareto_coeffs = torch.cat((self.pareto_coeffs,additional_columns),dim=1)
                
                self.pareto_coeffs = torch.cat((self.pareto_coeffs, coeff[s]))
            
            self.pareto_intercepts = torch.cat((self.pareto_intercepts,non_std_intercepts[s]))
            
            
            for comb in combinations_generated[s]:
                
                self.pareto_names.append(np.array(self.names)[comb.to(torch.int)].tolist())
                
           
        min_value, min_index = torch.min(mean, dim=0)
  
        coefs_min = torch.squeeze(sol[min_index]).unsqueeze(1)
        
        indices_min  = torch.squeeze(combinations_generated[min_index])
        
        non_std_coeff = ((coefs_min.T/self.x_std[indices_min.tolist()]))
        
        non_std_intercept = self.y.mean() - torch.dot(self.x_mean[indices_min.tolist()]/self.x_std[indices_min.tolist()],coefs_min.flatten())
        
        self.residual = self.y_centered - torch.mm(coefs_min.T,self.x_standardized[:,indices_min.tolist()].T)
        
        rmse = float(torch.sqrt(torch.mean(self.residual**2)))

        r2 = 1 - (float(torch.sum(self.residual**2))/float(torch.sum((self.y_centered)**2)))
        
        terms = []
        

        for i in range(len(non_std_coeff.squeeze())):
            
            ce = "{:.10f}".format(float(non_std_coeff.squeeze()[i]))
            
            term = str(ce) + "*" + str(self.names[int(indices_min[i])])
            
            
            terms.append(term)
            
        self.indices_clone = self.indices.clone()

        return float(rmse),terms,non_std_intercept,non_std_coeff,r2

    '''
    ##########################################################################################################################

    Defines the function to model the equation

    ##########################################################################################################################
    '''
    def regressor_fit(self):
        
        if self.x.shape[1] > self.sis_features*self.dimension:
            
            if self.disp:
                print()
                #print(f"Starting sparse model building in {self.device} \n")
            
        else:
            print('!!Important:: Given Number of features in SIS screening is greater than the feature space created, changing the SIS features to shape of features created!!')
            
            self.sis_features = self.x.shape[1]
            
            self.indices = torch.arange(1, (self.dimension*self.sis_features+1)).view(self.dimension*self.sis_features,1).to(self.device)
            
        #Looping over the dimensions 
        for i in range(1,self.dimension+1):
            
            if i ==1:
                
                start_1D = time.time()

                #calculate the scores
                scores = torch.abs(torch.mm(self.y_centered.unsqueeze(1).T,self.x_standardized))

                #Set the NaN values claculation to zero, instead of removing 
                scores[torch.isnan(scores)] = 0

                #Sort the top number of scores based on the sis_features 
                sorted_scores, sorted_indices = torch.topk(scores,k=self.sis_features)
                
                sorted_indices = sorted_indices.T
                
                remaining = torch.tensor((self.sis_features*self.dimension) - int(sorted_indices.shape[0])).to(self.device)

                #replace the remaining indices with nan
                nan = torch.full((remaining,1),float('nan')).to(self.device)
                
                sorted_indices = torch.cat((sorted_indices,nan),dim=0)
                
                #store the sorted indices as next column
                self.indices = torch.cat((self.indices,sorted_indices),dim=1)
                
                selected_index = self.indices[0,1]
                
                quantile_values = torch.quantile(self.complexity, torch.tensor(self.quantiles))
                '''
                
                try:
                    quantile_values = torch.quantile(self.complexity, torch.tensor(quantiles))
                    
                except:
                    print('Changing to manual partitions because of large tensor size..')
                
                    bins, digitized = (lambda t, n: (b := torch.arange(t.min(), t.max() + (w := (t.max() - t.min()) / n), w).add_(1e-6), torch.bucketize(t, b)))(self.complexity, len(quantiles)-1)
    
                    # The upper edges of the bins correspond to the quantiles
                    quantile_values = bins[1:]
                '''
                earlier_pareto_rmse = torch.empty(0,)
                
                earlier_pareto_complexity = torch.empty(0,)
                
                self.earlier_pareto_rmse = torch.cat((self.earlier_pareto_rmse,torch.sqrt(torch.mean(self.y_centered**2)).unsqueeze(0)),dim=0)
                
                self.earlier_pareto_complexity = torch.cat((self.earlier_pareto_complexity,torch.tensor([0.])))
                
                self.earlier_pareto_r2 = torch.cat((self.earlier_pareto_r2,torch.tensor([0.])),dim=0)
                
                self.pareto_names.extend([str(self.y_mean.tolist())])

                for i in range(len(quantile_values)):
                    
                    
                    if i == 0 : 
                        
                        ind = torch.where(self.complexity <= quantile_values[i])[0]
                        
                        scores1 = scores[:,ind]
                        
    
                    else: 
                        
                        ind = torch.where((self.complexity > quantile_values[i-1])&(self.complexity <= quantile_values[i]))[0]
                        
                        scores1 = scores[:,ind]

                    if self.quantiles[i] == 1.0: 
                        
                        ind = torch.where(self.complexity <= quantile_values[i])[0]
                        
                        scores1 = scores

                    if scores1.size()[1]==0: 
                        
                        continue
                
                    try:
                        
                        sorted_scores, sorted_indices = torch.topk(scores1,k=self.sis_features)
                        
                    except:
                        
                        sorted_scores, sorted_indices = torch.topk(scores1,k=len(scores1))
                
                    selected_indices = sorted_indices.flatten()

                    comp1 = self.complexity[ind]
                    
                    names = np.array(self.names)[ind]

                    x1 = self.x_standardized[:,selected_indices]
                    
                    x2 = x1.unsqueeze(0).T

                    y1 = self.y_centered.unsqueeze(1).unsqueeze(0)
                    
                    if x2.shape[0] != y1.shape[0]:
                        
                        y1 = y1.repeat(x2.shape[0],1,1)

                    has_nan_inf = torch.logical_or(
                        torch.isnan(x2).any(dim=1, keepdim=True).any(dim=2, keepdim=True),
                        torch.isinf(x2).any(dim=1, keepdim=True).any(dim=2, keepdim=True)
                        )
                    
                    x2 = torch.where(has_nan_inf,torch.zeros_like(x2),x2)
                    
                    try:
                        
                        sol,_,_,_ = torch.linalg.lstsq(x2,y1)
                        
                    except:
                        
                        x2_inv = torch.linalg.pinv(x2)
                        
                        sol = x2_inv@y1
                        
                        sol[torch.isnan(sol)] = 0
                    
                    
                    std = self.x_std[selected_indices].unsqueeze(0)
                    
                    non_std_sol = (sol/std)[:,0,0]
                    
                    xx = self.x_mean[selected_indices]
                    
                    yy = self.x_std[selected_indices]
                    
                    nn = xx/yy
                    
                    ss = nn*sol[:,0,0]
                    
                    non_std_intercepts = self.y.mean().repeat(len(ss)) - ss
                    
                    
                    predicted = torch.matmul(x2,sol)
                    
                    residuals = y1 - predicted
                    
                    square = torch.square(residuals)
                    
                    mean = torch.mean(square,dim=1,keepdim=True)
                    
                    features_rmse = torch.sqrt(mean)[:,0,0]
                    
                    features_r2 = 1 - (torch.sum(torch.square(residuals),dim=1)/torch.sum(torch.square(self.y_centered)))
                    
                    s= pareto(features_rmse,comp1[selected_indices]).pareto_front()
                    
                    
                    self.earlier_pareto_rmse = torch.cat((self.earlier_pareto_rmse,features_rmse[s]),dim=0)
                    
                    self.earlier_pareto_r2 = torch.cat((self.earlier_pareto_r2,features_r2[s].flatten()),dim=0)
                    
                    self.earlier_pareto_complexity = torch.cat((self.earlier_pareto_complexity,comp1[selected_indices[s]]))
                    
                    self.pareto_names.extend(np.array(self.names)[selected_indices.numpy()[s]].tolist())
                    
                    if non_std_sol[s].dim() ==1: 
                        
                        coeff_ad = non_std_sol[s].unsqueeze(1)
                        
                        
                    else:coeff_ad = non_std_sol[s]
                    
                    
                    self.pareto_coeffs = torch.cat((self.pareto_coeffs,coeff_ad))
                    
                    self.pareto_intercepts = torch.cat((self.pareto_intercepts,non_std_intercepts[s]))

                x_in = self.x[:, int(selected_index)].unsqueeze(1)

                # Add a column of ones to x for the bias term
                x_with_bias = torch.cat((torch.ones_like(x_in), x_in), dim=1).to(self.device)

                #Calculate the intercept and coefficient, Non standardized
                coef1, _, _, _ = torch.linalg.lstsq(x_with_bias, self.y)

                #Calculate the residuals based on the standardized and centered values
                x_in1 = self.x_standardized[:, int(selected_index)].unsqueeze(1)

                # Add a column of ones to x for the bias term
                x_with_bias1 = torch.cat((torch.ones_like(x_in1), x_in1), dim=1)
                
                coef, _, _, _ = torch.linalg.lstsq(x_with_bias1, self.y_centered)
                
                #pdb.set_trace()

                self.residual = (self.y_centered - (coef[1]*self.x_standardized[:, int(selected_index)])).unsqueeze(1).T
                
                rmse = float(torch.sqrt(torch.mean(self.residual**2)))
                
                r2 = 1 - (float(torch.sum(self.residual**2))/float(torch.sum((self.y_centered)**2)))
                
                coefficient = coef[1]

                intercept = self.y.mean() - torch.dot((self.x_mean[int(selected_index)]/self.x_std[int(selected_index)]).reshape(-1), coef[1].reshape(-1))#coef1[0]

                if intercept > 0:
                    
                    coefficient = coef[1]/self.x_std[int(selected_index)]
                    
                    coefficient = "{:.6f}".format(float(coefficient))
                    
                    equation = str(float(coefficient)) + '*' + str(self.names[int(selected_index)]) + '+' + str(float(intercept))
                    '''
                    if self.disp:
                        print('Equation: ', equation)
                        
                        print('\n')
                        
                        print('RMSE: ', rmse)
                        
                        print('R2::',r2)
                    '''
                    
                    
                else:
                    
                    coefficient = coef[1]/self.x_std[int(selected_index)]
                    
                    coefficient = "{:.6f}".format(float(coefficient))
                    
                    equation = str(float(coefficient)) + '*' + str(self.names[int(selected_index)])  + str(float(intercept))
                    '''
                    if self.disp:
                        print('Equation: ', equation)
                        
                        print('\n')
                        
                        print('RMSE: ', rmse)
                        
                        print('R2::',r2)
                    
                        print('Time taken to generate one dimensional equation: ', time.time()-start_1D,' seconds')
                        
                        print('\n')
                    '''
                
                if self.device == 'cuda':torch.cuda.empty_cache()
                
                if rmse <= self.rmse_metric and r2>= self.r2_metric: return float(rmse),equation,r2,self.earlier_pareto_rmse,self.earlier_pareto_complexity,self.pareto_names,self.pareto_intercepts,self.pareto_coeffs,self.earlier_pareto_r2

                
                
            else:
                
                start = time.time()
                
                self.indices_clone = self.indices.clone()

                rmse,terms,intercept,coefs,r2 = self.higher_dimension(i)
                
                equation =''
                
                for k in range(len(terms)):
                    
                    if coefs.flatten()[k] > 0:
                        
                        equation = equation + ' + ' + (str(terms[k]))+'  '
                        
                    else:
                        
                        equation = equation + (str(terms[k])) + '  '
                '''
                if self.disp:
                    print('Equation: ',equation[:len(equation)-1])
                    print('\n')
    
                    print('Intercept:', float(intercept))
                    print('\n')
    
                    print('RMSE:',float(rmse))
                    print('\n')
                    
                    print('R2::',r2)
    
                    print(f'Time taken for {i} dimension is: ', time.time()-start)
                '''
                #print('Intercept:',float(intercept))
                if self.device == 'cuda': torch.cuda.empty_cache()
                
                if rmse <= self.rmse_metric and r2>= self.r2_metric: 
                    print("Intercept:",float(intercept))
                    return float(rmse),equation,r2,self.earlier_pareto_rmse,self.earlier_pareto_complexity,self.pareto_names,self.pareto_intercepts,self.pareto_coeffs,self.earlier_pareto_r2

        return float(rmse),equation,r2,self.earlier_pareto_rmse,self.earlier_pareto_complexity,self.pareto_names,self.pareto_intercepts,self.pareto_coeffs,self.earlier_pareto_r2

