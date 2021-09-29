import pandas as pd
import matplotlib.pyplot as plt

df1=pd.read_csv('run-AR_0_hop-tag-AVTT_.csv')
df2=pd.read_csv('run-AR_1_hop-tag-AVTT_.csv')
df3=pd.read_csv('run-AR_2_hop-tag-AVTT_.csv')
df4=pd.read_csv('run-Q_routing-tag-AVTT_.csv')

fig, ax = plt.subplots()

ax.plot(df1.loc[:,'Value'],'lightblue')
ax.plot(df1.loc[:,'Value'].rolling(10).mean(),'b',label='AR(0hop)')

ax.plot(df2.loc[:,'Value'],'lightgreen')
ax.plot(df2.loc[:,'Value'].rolling(10).mean(),'g',label='AR(1hop)')

ax.plot(df3.loc[:,'Value'],'lightgrey')
ax.plot(df3.loc[:,'Value'].rolling(10).mean(),'k',label='AR(2hop)')

ax.plot(df4.loc[:,'Value'],'lightpink')
ax.plot(df4.loc[:,'Value'].rolling(10).mean(),'r',label='Q-routing')

ax.set_xlabel('episode number')
ax.set_ylabel('AVTT')
ax.set_title('D.T. Toronto Average Travel Time')
ax.legend()
# plt.show()

fig, ax = plt.subplots()

ax.plot(df1.loc[700:,'Value'],'lightblue')
ax.plot(df1.loc[700:,'Value'].rolling(10).mean(),'b',label='AR(0hop)')

ax.plot(df2.loc[700:,'Value'],'lightgreen')
ax.plot(df2.loc[700:,'Value'].rolling(10).mean(),'g',label='AR(1hop)')

ax.plot(df3.loc[700:,'Value'],'lightgrey')
ax.plot(df3.loc[700:,'Value'].rolling(10).mean(),'k',label='AR(2hop)')

ax.plot(df4.loc[700:,'Value'],'lightpink')
ax.plot(df4.loc[700:,'Value'].rolling(10).mean(),'r',label='Q-routing')

ax.set_xlabel('episode number')
ax.set_ylabel('AVTT')
ax.set_title('D.T. Toronto Average Travel Time')
ax.legend()


df1=pd.read_csv('run-AR_0_hop-tag-Routing Success_.csv')
df2=pd.read_csv('run-AR_1_hop-tag-Routing Success_.csv')
df3=pd.read_csv('run-AR_2_hop-tag-Routing Success_.csv')
df4=pd.read_csv('run-Q_routing-tag-Routing Success_.csv')

fig, ax = plt.subplots()

ax.plot(df1.loc[:,'Value'],'lightblue')
ax.plot(df1.loc[:,'Value'].rolling(10).mean(),'b',label='AR(0hop)')

ax.plot(df2.loc[:,'Value'],'lightgreen')
ax.plot(df2.loc[:,'Value'].rolling(10).mean(),'g',label='AR(1hop)')

ax.plot(df3.loc[:,'Value'],'lightgrey')
ax.plot(df3.loc[:,'Value'].rolling(10).mean(),'k',label='AR(2hop)')

ax.plot(df4.loc[:,'Value'],'lightpink')
ax.plot(df4.loc[:,'Value'].rolling(10).mean(),'r',label='Q-routing')

ax.set_xlabel('episode number')
ax.set_ylabel('AVTT')
ax.set_title('D.T. Toronto Average Travel Time')
ax.legend()

plt.show()