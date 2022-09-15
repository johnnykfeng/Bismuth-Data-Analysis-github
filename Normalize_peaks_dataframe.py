import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

peaksum_df = pd.read_sql_table('Peak_sum_23nm_adjusted',
                                'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project',
                               columns = ['scan_id', 'timepoint', 'peak1', 'peak2', 'peak5', 'peak6']
                               )

# peaksum_df = pd.read_csv('D:\\Bismuth Project\\Bismuth-Data-Analysis-github\\Peak_sum_23nm_adjusted.csv')

peaksum_df.sort_values(['scan_id','timepoint'], inplace=True)
peaksum_df.reset_index(drop=True, inplace=True)

exp_fit_23nm_df = pd.read_sql_table('exp_fit_variables_23nm',
                                    'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project',
                                    columns = ['scan_id', 'peak_id', 'a', 'tau', 't0', 'c'])

# print(peaksum_df.head())
# print(exp_fit_23nm_df.head())

scankeys = [1, 2, 3, 4, 5, 6, 7]
# scankeys = peaksum_df['scan_id'].unique()
# print(scankeys)

peaks = ['peak1', 'peak2', 'peak5', 'peak6']
peak_colors = cm.turbo(np.linspace(0, 1, len(peaks)))

big_peak_norm_list = []

for p, peak in enumerate(peaks):
    peak_norm_list = []
    print('p = ', p)
    print('peak = ', peak)

    for s in scankeys:
        print('scankey = ', s)


        df_scan = peaksum_df[peaksum_df['scan_id']==s].sort_values(['timepoint'])

        peak_values = df_scan[peak].values
        peak_norm = peak_values/np.mean(peak_values[:7]) # divide by the mean of the last 5 values

        peak_norm_list = np.append(peak_norm_list, peak_norm)

        # peak_norm_array = np.concatenate(peak_norm)
        # print(np.shape(peak_norm_array))

    # big_peak_norm_list.append(peak_norm_array)
    # peak_norm_array
    # for i in len(scankeys):
    #     peak_norm_array = np.append(peak_norm[i])

    big_peak_norm_list.append(peak_norm_list)

print(np.shape(big_peak_norm_list))

peak_norm_df = pd.DataFrame({'peak1_norm': big_peak_norm_list[0],
                            'peak2_norm': big_peak_norm_list[1],
                            'peak5_norm': big_peak_norm_list[2],
                            'peak6_norm': big_peak_norm_list[3]})



new_df = pd.concat([peaksum_df, peak_norm_df], axis = 1)
new_df.sort_values(['scan_id','timepoint'], inplace=True)

new_df.to_csv('D:\\Bismuth Project\\Bismuth-Data-Analysis-github\\Normalized_peaks.csv')


