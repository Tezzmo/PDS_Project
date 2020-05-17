#README
#Requires DE list from http://download.geonames.org/export/zip/

# %%
!pip install -e ..
# %%
import nextbike

# %%
df = nextbike.io.read_file()

# %%
df2 = nextbike.postalCodes.assignPostalCode(df)


# %%
map = nextbike.postalCodes.createTripsPerPostalCodeMap(df2,8,False) 
map
# %%
df2.columns


# %%
