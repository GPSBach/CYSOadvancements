'''
Read in Anonymized CYSO Data
Returns parsed/cleaned data for selected for fitting
	including one-hot encoding
Returns parsed/cleaned data without selection for data
	exploration including one-hot encoding
Returns parsed/cleaned data with county level aggregation
	for use with mapping software (Tableau)
'''

# imports
import pandas as pd
from fuzzywuzzy import process, fuzz
import re


# read in data
df = pd.read_csv('../data/CYSOanonymized.csv')

# set financial aid status to 0 if NaN

df['finaid'].fillna(0,inplace=True)

# parse race into different rows
df['race'].fillna('Other',inplace=True)
df['caucasian']=0
df['african_american']=0
df['native_american']=0
df['latino']=0
df['asian']=0
df['other_race']=0
df['race_num']=0

for index, row in df.iterrows():
    if re.search('cauc', str(row['race']), re.IGNORECASE):
        df.loc[index,'caucasian']=1
        df.loc[index,'race_num']=df.loc[index,'race_num']+1
    if re.search('african', str(row['race']), re.IGNORECASE):
        df.loc[index,'african_american']=1
        df.loc[index,'race_num']=df.loc[index,'race_num']+1
    if re.search('native', str(row['race']), re.IGNORECASE):
        df.loc[index,'caucasian']=1
        df.loc[index,'race_num']=df.loc[index,'race_num']+1
    if re.search('latino', str(row['race']), re.IGNORECASE):
        df.loc[index,'latino']=1
        df.loc[index,'race_num']=df.loc[index,'race_num']+1
    if re.search('asian', str(row['race']), re.IGNORECASE):
        if not re.search('caucasian', str(row['race']), re.IGNORECASE):
            df.loc[index,'asian']=1
            df.loc[index,'race_num']=df.loc[index,'race_num']+1
    if re.search('other', str(row['race']), re.IGNORECASE):
        df.loc[index,'other_race']=1
        df.loc[index,'race_num']=df.loc[index,'race_num']+1
    if re.search("`", str(row['race']), re.IGNORECASE):
        df.loc[index,'other_race']=1
        df.loc[index,'race_num']=df.loc[index,'race_num']+1

# compile gender
df['male']=0
df['female']=0

df.gender = df.gender.str.strip()
df.gender.replace('Male','M',inplace=True)
df.gender.replace('Female','F',inplace=True)

for index, row in df.iterrows():
    if row['gender']=='M':
        df.loc[index, 'male']=1
    if row['gender']=='F':
        df.loc[index,'female']=1

# parse major (1 if music major, 0 if non music major)
df['music']=0
music_terms = ['performance','music','violin','songwriting',
               'perfomance','bass','cello','viola','jazz']
for index, row in df.iterrows():
    for music_term in music_terms:
        if re.search(music_term,str(row['major']),re.IGNORECASE):
            df.loc[index,'music']=1
    

'''
map in and combine incomes
'''
#read in incomes and convert to numeric
income = pd.read_csv('../data/income_zip.csv')
df['homezip']=pd.to_numeric(df['homezip'], errors='coerce')

# merge with data structure
df = pd.merge(df,income,how='left',left_on='homezip',right_on='Zip')

# rename and reformat columns
df.rename(index=str, columns={"Mean": "mean_income", "Median": "median_income", \
                              "finaid" : "financial_aid", "race_num" : "multiracial"}, inplace=True)
df['mean_income'] = df['median_income'].str.replace(',','')
df['mean_income']=pd.to_numeric(df['mean_income'], errors='coerce')
df['median_income'] = df['median_income'].str.replace(',','')
df['median_income']=pd.to_numeric(df['median_income'], errors='coerce')
df['income_diff']=df['mean_income']-df['median_income']

'''
map in and combine college statistics data
'''

# read in college data
uf = pd.read_csv('../data/colleges.csv')

# filter to relevant universal columns
uf = uf[['displayName','acceptance-rate','institutionalControl']]

# build columns for merging and saving
df['uni_close'] = 'none'
df['uni_save'] = df['uni']

# replace strings

# strip name function
def stripnames(dataframe,instring,outstring):
    dataframe.loc[df['uni'].str.replace(' ','').replace("'",'').replace('-','') \
                  ==instring.replace(' ','').replace("'",'').replace('-',''),'uni']=outstring
    return dataframe

# strip and modify names for matching
uf['displayName'] = uf['displayName'].fillna('none')
df['uni'] = df['uni'].fillna('none given')
df['uni'] = df['uni'].str.replace('Uof','University of')
df['uni'] = df['uni'].str.replace('U of','University of')
df['uni'] = df['uni'].str.replace('Jacobs School of Music','')
df = stripnames(df,'Indiana University','Indiana University--Bloomington')
df = stripnames(df,'NYU','New York University')
df = stripnames(df,'Oberlin','Oberlin College')
df = stripnames(df,'UIC','University of Illinois Chicago')
df = stripnames(df,'UIUC','University of Illinois--Urbana Champaign')
df = stripnames(df,'USC(So.Cal.)','University of Southern California')
df = stripnames(df,'UofMichigan','University of Michigan')
df = stripnames(df,'Peabody','Johns Hopkins University')
df = stripnames(df,'University of I','University of Illinois--Urbana Champaign')
df = stripnames(df,'NIU','Northern Illinois University')
df = stripnames(df,'U.ofMinnesotaCarlsonSchoolofBusiness','University of Minnesota')
df = stripnames(df,'NEC','New England Conservatory')
df = stripnames(df,'MIT','Massachussetts Institute of Technology')
df = stripnames(df,'Gap Year','none')
df = stripnames(df,'Year off','none')
df = stripnames(df,'The Juilliard School','Julliard School')
df = stripnames(df,'IUBloomington,JacobsSchoolofMusic','Indiana University--Bloomington')
df = stripnames(df,'IUJacobsSchoolofMusic','Indiana University--Bloomington')
df = stripnames(df,'HarvardCollege','Harvard University')
df = stripnames(df,'Undecided','none')
df = stripnames(df,'NotgoingtoCollege','none')
df = stripnames(df,'IndianaUniversityJacobsSchoolofMusic','Indiana University--Bloomington')
df = stripnames(df,'IndianaUniversity,JacobsSchoolofMusic','Indiana University--Bloomington')
df = stripnames(df,'IndianaUniversity(JacobsSchoolofMusic)','Indiana University--Bloomington')
df = stripnames(df,'JacobsSchoolofMusic,atIndianaUniversity','Indiana University--Bloomington')
df = stripnames(df,'ClarendonHillsMiddleSchool','none')
df = stripnames(df,'IU','Indiana University--Bloomington')
df = stripnames(df,'LawrenceConservatory','Lawrence University')
df = stripnames(df,'IndianaUniversity','Indiana University--Bloomington')
df = stripnames(df,'WesternIllinoisUniversity','WesternIllinois')
df = stripnames(df,'UniversityofIllinois','University of Illinois--Urbana Champaign')
df = stripnames(df,',atIndianaUniversity','Indiana University--Bloomington')

# match name to closest
for index, row in df.iterrows():
    df.loc[index, 'uni_close']=process.extractOne(row['uni'],uf['displayName'])[0]

# merge university data into full data structure
df = pd.merge(df,uf,how='left',left_on='uni_close',right_on='displayName')

# encode university types
df['uni_public'] = 0
# df.loc(df['institutionalControl']=='private','institutionalControl')=0
# df.loc(df['institutionalControl']=='none','institutionalControl')=0
df.loc[df['institutionalControl']=='public','uni_public']=1

# drop spurious columns and rename others
df.drop(['Unnamed: 0','Zip','institutionalControl','displayName','uni_save','uni'],axis=1, inplace=True)
df.rename(index=str, columns={"uni_close": "uni","acceptance-rate":"acceptance_rate"}, inplace=True)

# force non numerics to numerics
df['acceptance_rate']=pd.to_numeric(df['acceptance_rate'], errors='coerce')

"""
Use zip codes to determine if in city of chicago
"""

# read in city of chicago zipcodes
zipcodes = pd.read_csv('../data/chicago_zipcodes.csv')

# convert to integers
df['s_zip']=pd.to_numeric(df['s_zip'], errors='coerce')
df.s_zip = df.s_zip.fillna(0)
df.s_zip = df.s_zip.astype(int)
df['homezip']=pd.to_numeric(df['homezip'], errors='coerce')
df.homezip = df.homezip.fillna(0)
df.homezip = df.homezip.astype(int)
zipcodes['ZIP']=pd.to_numeric(zipcodes['ZIP'], errors='coerce',downcast='integer')
zipcodes.ZIP = zipcodes.ZIP.astype(int)
df.loc[df['s_state'].isnull(),'s_zip'] = 60608

# convert states to be consistent
df.loc[df['s_state'].isnull(),'s_state'] = 'IL'
df.loc[df['s_state']=='Illinois','s_state']='IL'
df.loc[df['s_state']=='illinois','s_state']='IL'
df.loc[df['s_state']=='il','s_state']='IL'
df.loc[df['s_state']=='Il','s_state']='IL'
df.loc[df['s_state']=='iL','s_state']='IL'
df.loc[df['s_state']=='il.','s_state']='IL'
df.loc[df['s_state']=='Il ','s_state']='IL'
df.loc[df['s_state']=='Ilinois','s_state']='IL'
df.loc[df['s_state']=='Illinoid','s_state']='IL'
df.loc[df['s_state']=='Wisconsin','s_state']='WI'
df.loc[df['s_state']=='Ill','s_state']='IL'

# column for in chicago
df['chi_school']=0
df['notchi_school']=0
df['chi_home']=0
df['notchi_home']=0

df.loc[df['s_zip'].isin(zipcodes['ZIP']),'chi_school']=1
df.loc[~df['s_zip'].isin(zipcodes['ZIP']),'notchi_school']=1
df.loc[df['homezip'].isin(zipcodes['ZIP']),'chi_home']=1
df.loc[~df['homezip'].isin(zipcodes['ZIP']),'notchi_home']=1

"""
make seperate column for stratification?
"""

df['stratification_column'] = 0
df.loc[df['african_american']==1,'stratification_column']=1
df.loc[df['latino']==1,'stratification_column']=1

"""
add in public/private school status
make columns for public/private
make columns for chipub vs not chipub
"""

# read in high school information
hs = pd.read_csv('../data/highschools.csv')

#set up rows for matching
df['hs_close'] = 'none'
hs['school'] = hs['school'].fillna('none')
df['school'] = df['school'].fillna('none')
df['school'] = df['school'].str.replace(' ','').replace("'",'').replace('-','')
hs['school'] = hs['school'].str.replace(' ','').replace("'",'').replace('-','')

# match name to closest
for index, row in df.iterrows():
    df.loc[index, 'hs_close']=process.extractOne(row['school'],hs['school'])[0]
    
# merge data
df = pd.merge(df,hs,how='left',left_on='hs_close',right_on='school')

# make numeric column
df['hs_public'] = 0
df['hs_private'] = 0
df.loc[df['pph']=='Public','hs_public']=1
df.loc[df['pph']=='Private','hs_private']=1
df['chipub'] = df['chi_school']*df['hs_public']
df['subpub'] = df['notchi_school']*df['hs_public']

# save full data structure
df.to_csv('../data/exploring_data.csv')


# for fitting data including likely correlated features
df_fit = df[['female','other_race','notchi_school','notchi_home',
             'subpub','hs_public','chipub','hs_private',
             'uni_public','acceptance_rate','stratification_column',
             'music','male','caucasian','african_american',
             'latino', 'asian','chi_school','chi_home','financial_aid',
             'median_income']]

# for fitting data excluding likely correlated features
df_fit_limited = df[['female','other_race','notchi_school','notchi_home',
             'subpub','hs_public','chipub','hs_private',
             'stratification_column',
             'music','male','caucasian','african_american',
             'latino', 'asian','chi_school','chi_home','financial_aid']]


# save fitting data structure
df_fit.to_csv('../data/fitting_data.csv')
df_fit_limited.to_csv('../data/limited_data.csv')

# add column for aggregation to county level code
cf = pd.read_csv('../data/zip_county.txt')
cf = cf[['ZCTA5','COUNTY','STATE']]
df = pd.merge(df,cf,how='left',left_on='s_zip',right_on='ZCTA5')
df.loc[df['COUNTY'].isnull(),'COUNTY'] = 31
df.loc[df['STATE'].isnull(),'STATE'] = 17
df['s_county'] = '17031'
for index, row in df.iterrows():
    if row['COUNTY']>=100:
        df.loc[index,'s_county'] = str(int(row['STATE'])) + str(int(row['COUNTY']))
    else:
        df.loc[index,'s_county'] = str(int(row['STATE'])) + '0' + str(int(row['COUNTY']))
df.loc[df['s_county']=='42095','s_county']='42077'

# select data for mapping in tableau
df_map = df[['female','male','music','caucasian','african_american','hs_public','hs_private'
             'latino','asian','other_race','s_zip','s_state','s_county','financial_aid']]

#save to file
df_map.to_csv('../data/mapping_data_county.csv')

print('data cleaning finished')