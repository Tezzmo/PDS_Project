#%%
!pip install -e ..

# %%
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
from .utils import get_data_path
import nextbike
import os
# %%
df = nextbike.io.read_file()
df.info()


# %%
nnc = joblib.load('nearestNeighbor_PostalCode.pkl' , mmap_mode ='r')
toClassify = df[['p_lat','p_lng']]
toClassify

# %%
postalCodes = nnc.predict(toClassify)
dfPostalCodes = pd.DataFrame(postalCodes,columns=['p_postalCodes'])
dfPostalCodes
# %%
df['p_postalCodes'] = dfPostalCodes['p_postalCodes']

# %%
df

# %%
