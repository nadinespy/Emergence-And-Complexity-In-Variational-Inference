# %% plotting

# ----------------------------------------------------------------------------
# plots for phiid atoms & compositions
# ----------------------------------------------------------------------------

#for index, ax in enumerate(axs):

#    for j, weight in zip(range(0, len(all_weights)), all_weights):
#        temp_df = results_df.loc[((results_df.weight == weight),
#                                 quantities[index])]
#        plt.plot(all_rho, temp_df, '-', label = r'$\alpha$ = {:.2f}'.format(weight))
#        pylab.show()
#        plt.legend(ncol=2,loc=0)
#        #ax.axis([0,all_rho[-1],0,np.max(temp_model)])
        #ax.set_xlabel(r'$\rho$', fontsize=18)
        #ax.set_ylabel(r'$\varphi$', rotation=0, fontsize=18,labelpad=25)
#        title = r'$\varphi$ for different correlations & weights'


#plt.savefig(path_out2+r'discrete_steady_state_df_00_001_00_1_0_001.pdf', bbox_inches='tight')

#%%
   
# plots per correlation, error variance, and time-lag
for correlation in all_rho:
    for error_variance in all_errvar:
        for time_lag in all_time_lags:
            for weight in all_weights:
                for off_diag_covs in all_off_diag_covs:
            
                    # index for the phiid terms to be plotted, time for x-axis
                    time = np.arange(time_lag, T, 1).tolist()
                     
                    fig, axs = plt.subplots(4, 2, figsize=(8, 10))
                    fig.suptitle('rho = {}'.format(correlation)+', error variance = {}'.format(error_variance)+', time-lag = {}'.format(time_lag), fontsize = 10)
        
                    axs = axs.flatten()
                            
                    for index, ax in enumerate(axs):
                        
                        if case == 'continuous':
                            temp_model = results_df.loc[((results_df.correlation == correlation) & (results_df.error_variance == error_variance/np.sqrt(2/dt)) & (results_df.time_lag == time_lag)), phiid_terms[index]]
                        elif case == 'discrete':
                            temp_model = results_df.loc[((results_df.correlation == correlation) & (results_df.error_variance == error_variance) & (results_df.time_lag == time_lag)), phiid_terms[index]]
                            
                        # calculate moving average
                        moving_average_window = 120
                        moving_average = np.int(np.float(T/moving_average_window))                                         
                        moving_average_vector = np.array(range(0,len(temp_model),1))
                        moving_average_vector = moving_average_vector.astype(np.float64)
                        moving_average_vector.fill(np.nan) 
                        raw_vector = np.ma.array(temp_model, mask=np.isnan(temp_model))
                
                        for l in range(len(temp_model)):
                            if l < moving_average:
                                number_of_numbers_in_sum = np.count_nonzero(~np.isnan(raw_vector[l:l+moving_average]))
                                moving_average_vector[l] = np.sum(raw_vector[l:l+moving_average])/number_of_numbers_in_sum
                            elif l > (len(raw_vector)-moving_average):
                                number_of_numbers_in_sum = np.count_nonzero(~np.isnan(raw_vector[l-moving_average:l]))
                                moving_average_vector[l] = np.sum(raw_vector[l-moving_average:l])/number_of_numbers_in_sum
                            else:    
                                number_of_numbers_in_sum = np.count_nonzero(~np.isnan(raw_vector[l-moving_average:l+moving_average]))
                                moving_average_vector[l] = np.sum(raw_vector[l-moving_average:l+moving_average])/number_of_numbers_in_sum                     
                        
                        ax.plot(time, temp_model, label = phiid_terms[index], color = 'b', alpha = 0.6)
                        ax.plot(moving_average_vector, label ='moving-average', color = 'k', linewidth = 1)
                        ax.set_title(phiid_terms[index], color = 'r', pad=10)
                    fig.tight_layout()
                        
        
                    fig.savefig(path_out2 + 'phiid_quantities_' + 
                                str(correlation).replace('.', '') + '_' + 
                                str(error_variance).replace('.', '')  + '_' + 
                                str(time_lag) + '_' +
                                str(weight).replace('.', '') + '_' +
                                str(off_diag_covs).replace('.', '') + '_' + 
                                case + '_' +
                                str(gamma).replace('.', '') + 
                                '.png', dpi=300, bbox_inches='tight')  
                    plt.cla()
                    del fig
