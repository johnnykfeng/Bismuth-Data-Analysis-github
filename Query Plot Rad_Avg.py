import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.cm as cm
from scipy.signal import find_peaks
from scipy import interpolate
from matplotlib.font_manager import FontProperties
import time
import csv
from ScanDictionary import scanlist
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Boolean, \
    ARRAY, Numeric, Float, ForeignKey, ForeignKeyConstraint, column
from sqlalchemy import insert, update, select, inspect

# csv_direc = 'D:\\Bismuth Project\\Bismuth-Data-Analysis-github\\Big_Scan_14nm.csv'
# radavg_flat_df = pd.read_csv(csv_direc, index_col=[1], usecols = ['timepoint', 'scan_key', 'radavg_flattened'])
#
# # print(radavg_flat[:5])
#
# shortdata = radavg_flat_df[:1]
# radavg_flattened = shortdata.iloc[0]['radavg_flattened']
# # radavg_flattened = radavg_flattened.array
#
# # shortdata.plot(x = 'timepoint', y = 'radavg_flattened', kind = 'line')
# plt.plot(np.array(radavg_flattened))
# plt.show()

using_sql = 1
if using_sql:
    # sqlalchemy connection to postgresql database
    engine = create_engine('postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
    metadata = MetaData()
    connection = engine.connect()

    scandict = Table('scandictionary', metadata, autoload=True, autoload_with=engine, extend_existing=True)
    bigscan14nm = Table('Big_Scan_14nm', metadata, autoload=True, autoload_with=engine, extend_existing=True)

    peak_scan_list = []
    timepoint_list = []

    col = [bigscan14nm.c.radavg_flattened]

    stmt = select(col)
    stmt = stmt.where(bigscan14nm.c.scan_key == 11)
    result = connection.execute(stmt).fetchall()

plt.figure()

for i in np.arange(0,20,5):
    y = result[i][0]
    x = range(len(y))

    plt.plot(y)
plt.show()