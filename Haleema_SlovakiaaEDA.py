#!/usr/bin/env python
# coding: utf-8

# The main purpose of Exploratory Data Analysis is to help look at the data before making any assumption, The Covid-19 pandemic is the most crucial health disaster that has surrounded the world for the past year. The following data helps us to understand the consequences of the COVID‐19 outbreak.

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt


# read_csv is an important pandas function to read the csv files and do operations on it. and df.head() returns the 1st 5 rows. "owid_covid_data" is the data am working with.

# In[26]:


df = pd.read_csv("linear-comprehensive-covid-data.csv")
df.head()


# In[27]:


x = df.groupby(df.location)
Slovakia = x.get_group("Slovakia")


# In[28]:


Slovakia


# In[29]:


##We change the index number starting from 60,393-60,766 to 0-373 for easy understaning and readability.


# In[30]:


Slovakia=Slovakia.reset_index()
Slovakia.drop('index', axis=1, inplace=True)


# In[31]:


Slovakia


# In[32]:


##We now find the null values present in the given series of object. Using the 'isnull().sum()' function, one can find the number of missing values per coumn.


# In[33]:


Slovakia.isnull().sum()


# In[34]:


print(Slovakia.shape)


# In[35]:


##the Slovakia.shape tells that there are 222 rows and 41 columns in the DataFrame for the country Slovakia


# In[36]:


Slovakia.columns


# In[37]:


##We can also determine the null values in-terms of percentage


# In[38]:


features_with_na=[features for features in Slovakia.columns if Slovakia[features].isnull().sum()>1]
for feature in features_with_na:
    print(feature, np.round(Slovakia[feature].isnull().mean(), 4),  ' % missing values')


# In[39]:


Slovakia.info()


# In[18]:


##Using the heatmap, we can clearly locate the null values present in each column


# In[19]:


plt.figure(figsize=(25,5))
sns.heatmap(Slovakia.isnull(),yticklabels=False)
plt.title("Heatmap showing the null values present in dataframe-Slovakia_EDA")


# In[21]:


##From the above heatmap, we can see that there are many values near to 0 which means that there is no dependence between the occurence of missing values, As we can the null values present in few columns, and also few coulmns are entirely having null value, so we can drop those column using drop() function


# In[40]:


Slovakia.drop(columns=['icu_patients', 'icu_patients_per_million','weekly_icu_admissions','weekly_icu_admissions_per_million',
                       'weekly_icu_admissions_per_million','weekly_hosp_admissions_per_million',
                       'handwashing_facilities'], axis =1 ,inplace =True)
Slovakia.drop(columns=['weekly_hosp_admissions'], axis =1 ,inplace =True)


# In[41]:


plt.figure(figsize=(25,5))
sns.heatmap(Slovakia.isnull(),yticklabels=False)
plt.title("Heatmap showing the null values present in dataframe-Slovakia_EDA")


# In[24]:


##1. Distrubution plot for total vaccinations provided


# In[42]:


sns.set(rc={'figure.figsize':(6,3)})
sns.distplot(Slovakia.total_vaccinations)
Slovakia['total_vaccinations'].describe()


# In[43]:


##2. Distrubution plot showing the number of people vaccinated


# In[44]:


sns.set(rc={'figure.figsize':(6,3)})
sns.distplot(Slovakia.people_vaccinated)
Slovakia['people_vaccinated'].describe()


# In[45]:


##3. Distrubution plot for total vaccinations provided


# In[46]:


sns.set(rc={'figure.figsize':(6,3)})
sns.distplot(Slovakia.total_vaccinations)
Slovakia['total_vaccinations'].describe()


# In[47]:


##4. Distrubution plot shows that people who are fully vaccinated


# In[48]:


sns.set(rc={'figure.figsize':(6,3)})
sns.distplot(Slovakia.people_fully_vaccinated)
Slovakia['people_fully_vaccinated'].describe()


# In[49]:


##5. Distrubution plot for the new_vaccinations available


# In[50]:


sns.set(rc={'figure.figsize':(6,3)})
sns.distplot(Slovakia.new_vaccinations)
Slovakia['new_vaccinations'].describe()


# In[51]:


sns.set(rc={'figure.figsize':(6,3)})
sns.distplot(Slovakia.new_vaccinations_smoothed)
Slovakia['new_vaccinations_smoothed'].describe()


# In[52]:


##6. Distrubution plot for total_vaccinations_per_hundred


# In[53]:


sns.set(rc={'figure.figsize':(6,3)})
sns.distplot(Slovakia.total_vaccinations_per_hundred)
Slovakia['total_vaccinations_per_hundred'].describe()


# In[54]:


sns.set(rc={'figure.figsize':(6,3)})
sns.distplot(Slovakia.people_vaccinated_per_hundred)
Slovakia['people_vaccinated_per_hundred'].describe()


# In[55]:


##8. Distrubution plot for people_fully_vaccinated_per_hundred


# In[56]:


sns.set(rc={'figure.figsize':(6,3)})
sns.distplot(Slovakia.people_fully_vaccinated_per_hundred)
Slovakia['people_fully_vaccinated_per_hundred'].describe()


# In[57]:


##9. Distrubution plot for new_vaccinations_smoothed_per_million


# In[58]:


sns.set(rc={'figure.figsize':(6,3)})
sns.distplot(Slovakia.new_vaccinations_smoothed_per_million)
Slovakia['new_vaccinations_smoothed_per_million'].describe()


# In[59]:


Slovakia.describe().loc[:,['total_vaccinations','people_vaccinated','people_fully_vaccinated','new_vaccinations',
                           'new_vaccinations_smoothed','total_vaccinations_per_hundred','people_vaccinated_per_hundred',
                           'people_fully_vaccinated_per_hundred','new_vaccinations_smoothed_per_million']]


# In[60]:


##The following heatmap shows that its the correlation heatmap representing the country Slovakia. The correlation matrix contains the number of vaccinations which were available and how many people are vaccinated partially and how many people are fully vaccinated


# In[61]:


Slovakia_corr = Slovakia[['total_vaccinations','people_vaccinated','people_fully_vaccinated','new_vaccinations',
                           'new_vaccinations_smoothed','total_vaccinations_per_hundred','people_vaccinated_per_hundred',
                           'people_fully_vaccinated_per_hundred','new_vaccinations_smoothed_per_million']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(Slovakia_corr,annot = True,vmin =1,vmax=1,cmap ='GnBu_r')
plt.title("Correlation Matrix of Slovakia_EDA")
Slovakia_corr


# In[62]:


##These coulmns are filled with the mean values so as to remove the null value from the respective columns


# In[63]:


Slovakia['total_vaccinations']=Slovakia['total_vaccinations'].fillna(Slovakia['total_vaccinations'].mean())
Slovakia['people_vaccinated']=Slovakia['people_vaccinated'].fillna(Slovakia['people_vaccinated'].mean())

Slovakia['people_fully_vaccinated']=Slovakia['people_fully_vaccinated'].fillna(Slovakia['people_fully_vaccinated'].mean())
Slovakia['new_vaccinationsnew_vaccinations']=Slovakia['new_vaccinations'].fillna(Slovakia['new_vaccinations'].mean())

Slovakia['new_vaccinations_smoothed']=Slovakia['new_vaccinations_smoothed'].fillna(Slovakia['new_vaccinations_smoothed'].mean())
Slovakia['total_vaccinations_per_hundred']=Slovakia['total_vaccinations_per_hundred'].fillna(Slovakia['total_vaccinations_per_hundred'].mean())

Slovakia['people_vaccinated_per_hundred']=Slovakia['people_vaccinated_per_hundred'].fillna(Slovakia['people_vaccinated_per_hundred'].mean())
Slovakia['people_fully_vaccinated_per_hundred']=Slovakia['people_fully_vaccinated_per_hundred'].fillna(Slovakia['people_fully_vaccinated_per_hundred'].mean())

Slovakia['new_vaccinations']=Slovakia['new_vaccinations'].fillna(Slovakia['new_vaccinations'].mean())
Slovakia['new_vaccinations_smoothed_per_million']=Slovakia['new_vaccinations_smoothed_per_million'].fillna(Slovakia['new_vaccinations_smoothed_per_million'].mean())


# In[64]:



plt.figure(figsize=(25,5))
sns.heatmap(Slovakia.isnull(),yticklabels=False)
plt.title("Heatmap showing the null values present in dataframe-Slovakia_EDA")


# In[65]:


##we use describe function to view the stastical details like percentile,mean etc of a dataframe.


# In[66]:


Slovakia.describe().loc[:,['new_cases_smoothed','total_deaths','new_deaths','new_deaths_smoothed','new_cases_smoothed_per_million',
                           'new_deaths_per_million','total_deaths_per_million','new_deaths_smoothed_per_million','reproduction_rate',
                          'hosp_patients','hosp_patients_per_million','new_tests_smoothed','new_tests_smoothed_per_thousand',
                           'positive_rate','tests_per_case']]


# In[67]:


##The following correlation matrix contains the number of deaths that took place and the positive rate and the number of test per case


# In[68]:


Slovakia_corr = Slovakia[['new_cases_smoothed','total_deaths','new_deaths','new_deaths_smoothed','new_cases_smoothed_per_million',
                           'new_deaths_per_million','total_deaths_per_million','new_deaths_smoothed_per_million','reproduction_rate',
                          'hosp_patients','hosp_patients_per_million','new_tests_smoothed','new_tests_smoothed_per_thousand',
                           'positive_rate','tests_per_case']].corr()
plt.figure(figsize=(15,10))
sns.heatmap(Slovakia_corr,annot = True,vmin =1,vmax=1,cmap ='GnBu_r')
plt.title("Correlation Matrix of Slovakia_EDA")
Slovakia_corr


# In[69]:


##These coulmns are filled with the mean values so as to remove the null value from the respective columns


# In[70]:


Slovakia['new_cases_smoothed']=Slovakia['new_cases_smoothed'].fillna(Slovakia['new_cases_smoothed'].mean())
Slovakia['total_deaths']=Slovakia['total_deaths'].fillna(Slovakia['total_deaths'].mean())

Slovakia['new_deaths']=Slovakia['new_deaths'].fillna(Slovakia['new_deaths'].mean())
Slovakia['new_deaths_smoothed']=Slovakia['new_deaths_smoothed'].fillna(Slovakia['new_deaths_smoothed'].mean())

Slovakia['new_cases_smoothed_per_million']=Slovakia['new_cases_smoothed_per_million'].fillna(Slovakia['new_cases_smoothed_per_million'].mean())
Slovakia['new_deaths_per_million']=Slovakia['new_deaths_per_million'].fillna(Slovakia['new_deaths_per_million'].mean())

Slovakia['total_deaths_per_million']=Slovakia['total_deaths_per_million'].fillna(Slovakia['total_deaths_per_million'].mean())
Slovakia['new_deaths_smoothed_per_million']=Slovakia['new_deaths_smoothed_per_million'].fillna(Slovakia['new_deaths_smoothed_per_million'].mean())

Slovakia['reproduction_rate']=Slovakia['reproduction_rate'].fillna(Slovakia['reproduction_rate'].mean())
Slovakia['hosp_patients']=Slovakia['hosp_patients'].fillna(Slovakia['hosp_patients'].mean())

Slovakia['hosp_patients_per_million']=Slovakia['hosp_patients_per_million'].fillna(Slovakia['hosp_patients_per_million'].mean())
Slovakia['new_tests_smoothed']=Slovakia['new_tests_smoothed'].fillna(Slovakia['new_tests_smoothed'].mean())

Slovakia['new_tests_smoothed_per_thousand']=Slovakia['new_tests_smoothed_per_thousand'].fillna(Slovakia['new_tests_smoothed_per_thousand'].mean())
Slovakia['positive_rate']=Slovakia['positive_rate'].fillna(Slovakia['positive_rate'].median())
Slovakia['tests_per_case']=Slovakia['tests_per_case'].fillna(Slovakia['tests_per_case'].mean())

Slovakia['stringency_index']=Slovakia['stringency_index'].fillna(Slovakia['stringency_index'].mean())
Slovakia['new_tests']=Slovakia['new_tests'].fillna(Slovakia['new_tests'].mean())


# In[71]:


plt.figure(figsize=(25,5))
sns.heatmap(Slovakia.isnull(),yticklabels=False)
plt.title("Heatmap showing the null values present in dataframe-Slovakia_EDA")


# In[72]:


##The function info() tells us that the number of null values and number of non-null values present in the dataframe


# In[73]:


Slovakia.info()


# In[74]:


##As we can observe in the above table, that the date in the given dataframe is of the type object. So converting it into data_time object


# In[75]:


Slovakia['date'] = pd.to_datetime(Slovakia['date'],format='%Y-%m-%d')


# In[76]:


print("Starting date =",Slovakia['date'].min())
print("End date =",Slovakia['date'].max())
print("Length of data with respect to days :",Slovakia['date'].max()-Slovakia['date'].min())


# In[77]:


##So we have the Slovakia _country data for 373 days that is from 6th march 2020 (2020-03-06) to 14th March 2020 (2020-03-14)
##Now we can obsverve the given data interms of line_plot which shows the clear representation of that particular feature with respect to that particular date
##1. plot showing the number of total cases with respect to date


# In[78]:


plt.figure(figsize=(10,6))
plt.plot(Slovakia['date'],Slovakia['total_cases'])
plt.title('total_cases')
plt.tick_params(axis='x', rotation=0)


# In[79]:


##2. plot showing the number of new cases with respect to date


# In[80]:


plt.figure(figsize=(10,6))
plt.plot(Slovakia['date'],Slovakia['new_cases'])
plt.title('new_cases')
plt.tick_params(axis='x', rotation=0)


# In[81]:


##3. plot showing the number of new_cases_smoothed with respect to date


# In[82]:


plt.figure(figsize=(10,6))
plt.plot(Slovakia['date'],Slovakia['new_cases_smoothed'])
plt.title('new_cases_smoothed')
plt.tick_params(axis='x', rotation=0)


# In[83]:


##4. plot showing the number of total_deaths with respect to date


# In[84]:


plt.figure(figsize=(10,6))
plt.plot(Slovakia['date'],Slovakia['total_deaths'])
plt.title('total_deaths')
plt.tick_params(axis='x', rotation=0)


# In[85]:


##5. plot showing the number of new_deaths with respect to date


# In[86]:


plt.figure(figsize=(10,6))
plt.plot(Slovakia['date'],Slovakia['new_deaths'])
plt.title('new_deaths')
plt.tick_params(axis='x', rotation=0)


# In[87]:


##6. plot showing the number of new_deaths_smoothed with respect to date


# In[88]:


plt.figure(figsize=(10,6))
plt.plot(Slovakia['date'],Slovakia['new_deaths_smoothed'])
plt.title('new_deaths_smoothed')
plt.tick_params(axis='x', rotation=0)


# In[89]:


##7. plot showing the number of total_deaths_per_million with respect to date


# In[90]:


plt.figure(figsize=(10,6))
plt.plot(Slovakia['date'],Slovakia['total_deaths_per_million'])
plt.title('total_deaths_per_million')
plt.tick_params(axis='x', rotation=0)


# In[91]:


##8. plot showing the number of new_deaths_per_million with respect to date


# In[92]:


plt.figure(figsize=(10,6))
plt.plot(Slovakia['date'],Slovakia['new_deaths_per_million'])
plt.title('new_deaths_per_million')
plt.tick_params(axis='x', rotation=0)


# In[93]:


##9. plot showing the number of new_deaths_smoothed_per_million with respect to date


# In[94]:


plt.figure(figsize=(10,6))
plt.plot(Slovakia['date'],Slovakia['new_deaths_smoothed_per_million'])
plt.title('new_deaths_smoothed_per_million')
plt.tick_params(axis='x', rotation=0)


# In[95]:


##10. plot showing the number of new_tests with respect to date


# In[96]:


plt.figure(figsize=(10,6))
plt.plot(Slovakia['date'],Slovakia['new_tests'])
plt.title('new_tests')
plt.tick_params(axis='x', rotation=0)


# In[97]:


##11. plot showing the number of total_tests with respect to date


# In[98]:


plt.figure(figsize=(10,6))
plt.plot(Slovakia['date'],Slovakia['total_tests'])
plt.title('total_tests')
plt.tick_params(axis='x', rotation=0)


# In[99]:


##12. plot showing the number of positive_rate with respect to date


# In[100]:


plt.figure(figsize=(10,6))
plt.plot(Slovakia['date'],Slovakia['positive_rate'])
plt.title('positive_rate')
plt.tick_params(axis='x', rotation=0)


# In[101]:


##13. plot showing the number of stringency_index with respect to date


# In[102]:


plt.figure(figsize=(10,6))
plt.plot(Slovakia['date'],Slovakia['stringency_index'])
plt.title('stringency_index')
plt.tick_params(axis='x', rotation=0)


# In[103]:


##Adding a new feature 'month' to findout the requirements with respect to each month


# In[104]:


Slovakia['month'] = Slovakia['date'].dt.month_name()


# In[105]:


Slovakia.columns


# In[106]:


Slovakia_new = Slovakia.loc[:,'date':'stringency_index']


# In[107]:


##Now, we group the gven data in months, so that we can find out the features with respect to each month


# In[108]:


Slovakiaa = Slovakia.groupby(Slovakia.month).sum()
Slovakiaa = Slovakiaa.reindex(['April','May','June','July','August','September','October','November','December',
             'January', 'February','March'])
Slovakiaa = Slovakiaa.loc[:,['total_cases','new_cases','new_cases_smoothed','total_deaths','new_deaths','new_deaths_smoothed','total_cases_per_million',
                             'new_cases_per_million','new_cases_smoothed_per_million','total_deaths_per_million','new_deaths_per_million',
                             'new_deaths_smoothed_per_million','total_tests','positive_rate','tests_per_case','total_vaccinations',
                             'people_vaccinated','stringency_index']]


# In[109]:


##The following table tells the variation of each feature with repect to the month
Slovakiaa


# In[110]:


##so the above table shows that all features in the given data are grouped according to respective months. We can see that the total cases for each month is increasing rapidly, and even the number of new cases and number of deaths that are taking place each month is increasing


# In[111]:


Slovakia.index


# In[112]:


##The following plots gives us the clear picture of each feature of the dataframe with respect to that particular month for country Slovakia (2020-2021)
##1. line_plot showing the number of total cases for each month


# In[113]:


plt.figure(figsize=(12,6))
g = sns.lineplot(x=Slovakiaa.index,y=Slovakiaa.total_cases,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("total_cases vs month  for germany ")


# In[114]:


##2. line_plot showing the number of new cases for each month


# In[115]:


plt.figure(figsize=(12,6))
g = sns.lineplot(x=Slovakiaa.index,y=Slovakiaa.new_cases,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("new_cases vs month for germany ")


# In[116]:


##3. line_plot showing the number of new_cases_smoothed for each month


# In[117]:


plt.figure(figsize=(12,6))
g = sns.lineplot(x=Slovakiaa.index,y=Slovakiaa.new_cases_smoothed,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("new_cases_smoothed vs month  for germany ")


# In[118]:


##4. line_plot showing the number of total_deaths for each month


# In[119]:


plt.figure(figsize=(12,6))
g = sns.lineplot(x=Slovakiaa.index,y=Slovakiaa.total_deaths,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("total_deaths vs month  for germany ")


# In[120]:


##5. line_plot showing the number of new_deaths for each month


# In[121]:


plt.figure(figsize=(12,6))
g = sns.lineplot(x=Slovakiaa.index,y=Slovakiaa.new_deaths,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("new_deaths vs month  for germany ")


# In[122]:


##6. line_plot showing the number of new_deaths_smoothed for each month


# In[123]:


plt.figure(figsize=(12,6))
g = sns.lineplot(x=Slovakiaa.index,y=Slovakiaa.new_deaths_smoothed,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("new_deaths_smoothed vs month  for germany ")


# In[124]:


##7. line_plot showing the number of total_cases_per_million for each month


# In[125]:


plt.figure(figsize=(12,6))
g = sns.lineplot(x=Slovakiaa.index,y=Slovakiaa.total_cases_per_million,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("total_cases_per_million vs month  for germany ")


# In[126]:


##8. line_plot showing the number of new_cases_per_million for each month


# In[127]:


plt.figure(figsize=(12,6))
g = sns.lineplot(x=Slovakiaa.index,y=Slovakiaa.new_cases_per_million,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("new_cases_per_million vs month  for germany ")


# In[128]:


##9. line_plot showing the number of total_deaths_per_million for each month


# In[129]:


plt.figure(figsize=(12,6))
g = sns.lineplot(x=Slovakiaa.index,y=Slovakiaa.total_deaths_per_million,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("total_deaths_per_million vs month  for germany ")


# In[130]:


##10. line_plot showing the number of total_tests for each month


# In[131]:


plt.figure(figsize=(12,6))
g = sns.lineplot(x=Slovakiaa.index,y=Slovakiaa.total_tests,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("total_tests vs month  for germany ")


# In[132]:


##11. line_plot showing the number of positive_rate for each month


# In[133]:


plt.figure(figsize=(12,6))
g = sns.lineplot(x=Slovakiaa.index,y=Slovakiaa.positive_rate,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("positive_rate vs month  for germany ")


# In[134]:


##12. line_plot showing the number of stringency_index for each month
plt.figure(figsize=(12,6))
g = sns.lineplot(x=Slovakiaa.index,y=Slovakiaa.stringency_index,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("stringency_index vs month  for germany ")


# In[135]:


##The above lineplots shows the variation of cases for each month. We can also plot the same using bar_plot for better understanding for country Slovakia (2020-2021)


# # 1. bar_plot showing the number of total_cases for each month

# In[138]:


plt.figure(figsize=(12,6))
g = sns.barplot(x=Slovakiaa.index,y=Slovakiaa.total_cases,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("total_cases vs month  for germany ")


# # 2. bar_plot showing the number of new_cases for each month

# In[137]:


plt.figure(figsize=(12,6))
g = sns.barplot(x=Slovakiaa.index,y=Slovakiaa.new_cases,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("new_cases vs month  for germany ")


# # 3. bar_plot showing the number of total_deaths for each month
# 

# In[139]:


plt.figure(figsize=(12,6))
g = sns.barplot(x=Slovakiaa.index,y=Slovakiaa.total_deaths,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("total_deaths vs month  for germany ")


# # 4. bar_plot showing the number of new_deaths for each month
# 

# In[140]:


plt.figure(figsize=(12,6))
g = sns.barplot(x=Slovakiaa.index,y=Slovakiaa.new_deaths,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("new_deaths vs month  for germany ")


# #  5. bar_plot showing the number of total_cases_per_million for each month

# In[141]:


plt.figure(figsize=(12,6))
g = sns.barplot(x=Slovakiaa.index,y=Slovakiaa.total_cases_per_million,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("total_cases_per_million vs month  for germany ")


# # 6. bar_plot showing the number of new_cases_per_million for each month
# 

# In[143]:


plt.figure(figsize=(12,6))
g = sns.barplot(x=Slovakiaa.index,y=Slovakiaa.new_cases_per_million,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("new_cases_per_million vs month  for germany ")


# # 7. bar_plot showing the number of total_deaths_per_million for each month

# In[145]:


plt.figure(figsize=(12,6))
g = sns.barplot(x=Slovakiaa.index,y=Slovakiaa.total_deaths_per_million,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("total_deaths_per_million vs month  for germany ")


# # 8. bar_plot showing the number of total_tests for each month

# In[146]:


plt.figure(figsize=(12,6))
g = sns.barplot(x=Slovakiaa.index,y=Slovakiaa.total_tests,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("total_tests vs month  for germany ")


# # 9. bar_plot showing the number of positive_rate for each month

# In[147]:


plt.figure(figsize=(12,6))
g = sns.barplot(x=Slovakiaa.index,y=Slovakiaa.positive_rate,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("positive_rate vs month  for germany ")


# # 10. bar_plot showing the number of stringency_index for each month

# In[148]:


plt.figure(figsize=(12,6))
g = sns.barplot(x=Slovakiaa.index,y=Slovakiaa.stringency_index,data=Slovakiaa)
g.set_xticklabels(Slovakiaa.index,rotation=45)
g.set_title("stringency_index vs month for germany ")


# In[ ]:




