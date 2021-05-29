#--------------priyantha appuhamy----------------------
import pandas as pd
import numpy as np
#import math
import os
import datetime
import re
import sys
from datetime import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
#import plotly.graph_objs as go

#----------------------------------------
#read csv files in the folder
filenames=[]
pro_nam =[]
file = ""

#writing to csv
data_out = []

pro_pla = []
pro_act = []
pro_dat = []

data_out_pla_init = []
data_out_pla = np.array(data_out_pla_init)
data_out_act_init = []
data_out_act = np.array(data_out_act_init)

step_count = 50 #step*step_count total time expand of the progress
step = 5 #in days

df_pro_info = pd.DataFrame()
pro_df_raw = pd.DataFrame()
pro_df_fin = pd.DataFrame()

#basepath = os.path.abspath(".")

path_in = r"C:\Users\Priyantha\codes\Proj_dc\projects_in"
path_out = r"C:\Users\Priyantha\codes\Proj_dc\projects_out"
path_info = r"C:\Users\Priyantha\codes\Proj_dc\projects_info"
path_in_fin = r"C:\Users\Priyantha\codes\Proj_dc\projects_fin"


# path_in = r"home/priyanthaappuhamy/mysite/projects_in"
# path_out = r"home/priyanthaappuhamy/mysite/projects_out"
# path_info = r"home/priyanthaappuhamy/mysite/projects_info"
# path_in_fin = r"home/priyanthaappuhamy/mysite/projects_fin"

#----------------get the project names------------------
for root,dirs,files in os.walk(path_info):
    for file in files:
        if file.endswith(".csv"):
            filenames.append(os.path.join(path_info,file))
    if filenames==[]:
        print("No files found in the directory")
        sys.exit()

for index,filename in enumerate(filenames):
    df_pro_info = pd.read_csv(filename)

filenames = []

# #---------------------getting the current date time user input-------------

# #cur_date_time =  input("Enter the Progress reporting date in MM/DD/YY HH:MM CC format e.g.'01/31/21 08:00 AM':")
# #cur_date_time = '04/19/21 08:00 AM'

now = datetime.now()
cur_date_time = now.strftime("%m/%d/%y %I:%M %p")

# #---------------------------------------------------------get progress info of each project ---------------------

for root,dirs,files in os.walk(path_in):
    for file in files:
        if file.endswith(".csv"):
            filenames.append(os.path.join(path_in,file))
    if filenames==[]:
        print("No files found in the directory")
        sys.exit()

#for all the files in the directory
for index,filename in enumerate(filenames):
    pro_df_raw = pd.read_csv(filename)
    #remove the tasks not representing progress summery etc.
    drop_index = []
    for i in range(0,len(pro_df_raw['Name']),1):
        if pro_df_raw['S_Task'][i] ==1 or pro_df_raw['S_Task'][i] =='1':
           drop_index.extend([i])
        else:
           pass
    pro_df = pro_df_raw.drop(drop_index)


    #function check whether NaN is there
    def isNaN(num):
        return num != num

    # funtion to convert date string to date
    import datetime
    def date_time_to_date(date_time_str):
        if isNaN(date_time_str):
            return 0
        else:
            date_time_obj = datetime.datetime.strptime(date_time_str, '%m/%d/%y %I:%M %p')
            return (date_time_obj.date())

    # convert string to date format
    date_B_Finish = [date_time_to_date(i) for i in pro_df['B_Finish']]
    date_B_Start = [date_time_to_date(i) for i in pro_df['B_Start']]
    date_A_Finish=[date_time_to_date(i) for i in pro_df['A_Finish']]
    date_A_Start=[date_time_to_date(i) for i in pro_df['A_Start']]

    #taking no of days differnce between start date and current date of basedlines tasks
    X = [(date_B_Finish[i]-date_B_Start[i]).days if date_B_Finish[i]!=date_B_Start[i]  else 1 for i in range(0,len(date_B_Finish),1)]

    #progress measurement date
    cur_day = date_time_to_date(cur_date_time)

    #check the % complete for the each task with compared to baseline start and finish
    progress_task_pla = []
    progress_task_act = []

    #funtion to get the planned progress
    def get_task_pla_com(req_date):
        a = np.zeros(len(date_B_Finish))
        cur_day = date_time_to_date(req_date)
        for i in range(0,len(date_B_Finish),1):
            if date_B_Start[i]==0:
                a[i]=0
            else:
                a[i]= ((cur_day-date_B_Start[i]).days+1)/((date_B_Finish[i]-date_B_Start[i]).days+1)
                if a[i]>0 and a[i]<1:
                    progress_task_pla.extend([i])
                elif a[i]<0:
                    a[i]=0
                elif a[i]>1:
                    a[i]=1
                else:
                    pass
        return a


    #check the percent complete actuals

    def get_task_act_com(req_date):
        b = np.zeros(len(date_B_Finish))
        cur_day = date_time_to_date(req_date)
        for i in range(0,len(date_B_Finish),1):
            if date_A_Start[i]==0:
                b[i]=0
            elif cur_day>=date_A_Start[i]:
                if date_A_Finish[i]==0:
                    b[i]= pro_df['P_Com'].iloc[[i]]
                    progress_task_act.extend([i])
                else:
                    b[i]=1
            else:
                pass
        return b


    # get the number of progress values
    for date_fre in range(0,step_count+1,1):
        new_date = (date_time_to_date(cur_date_time) - datetime.timedelta((step_count-date_fre)*step)).strftime('%m/%d/%y %I:%M %p')
        progress_planned = round(100*get_task_pla_com(new_date).dot(X)/sum(X),2)
        progress_actual = round(100*get_task_act_com(new_date).dot(X)/sum(X),2)

        #data obtain
        pro_pla.extend([progress_planned])
        pro_act.extend([progress_actual])
        pro_dat.extend([new_date])

    #converting pla and act data to numpy array

    data_out_pla =np.array(pro_pla)
    data_out_act =np.array(pro_act)

    #get the date points
    pro_dat_ext = pro_dat[0:step_count+1]

    #get the project name
    pro_nam.extend([re.findall(r'\\(\w+)', filename)[-1]])

filenames = []

#----------------------------------------------------------------------------------------------

data_out_pla = data_out_pla.reshape(len(pro_nam),step_count+1)
data_out_act = data_out_act.reshape(len(pro_nam),step_count+1)


#get the output thorugh dataframe to csv
df_op_1 = pd.DataFrame(data=data_out_pla)
df_op_2 = pd.DataFrame(data=data_out_act)

#reshape for the project in each column
df_pla = df_op_1.T
df_act = df_op_2.T

#add header raw
df_pla.columns = pro_nam
df_act.columns = pro_nam

#----------------exporting project progress details----

##writing output to csv file
# export_file_path = path_out +"/"+"projects_pla.csv"
# df_pla.to_csv(export_file_path, index=False, header=True)
# export_file_path = path_out +"/"+"projects_act.csv"
# df_act.to_csv(export_file_path, index=False, header=True)

#---------------project finance---------------------
# #---------------------------------financial details-----------------

pro_fin_dic= {}

#------------------------get financial details of each project-----------------------

for root,dirs,files in os.walk(path_in_fin):
    for file in files:
        if file.endswith(".csv"):
            filenames.append(os.path.join(path_in_fin,file))
    if filenames==[]:
        print("No files found in the directory")
        sys.exit()

#for all the files in the directory
for index,filename in enumerate(filenames):

    #get the project name
    pro_nam_fin = re.findall(r'\\(\w+)', filename)[-1]

    #get the project finance details
    pro_df_fin = pd.read_csv(filename)

    #fill nul values
    pro_df_fin['P_value_LKR']=pro_df_fin['P_value_LKR'].fillna(0)
    pro_df_fin['A_value_LKR']=pro_df_fin['A_value_LKR'].fillna(0)

    #get the cumulative values
    pro_df_fin['P_value_LKR_cum'] = pro_df_fin['P_value_LKR'].cumsum()
    pro_df_fin['A_value_LKR_cum'] = pro_df_fin['A_value_LKR'].cumsum()

    #plot time variable
    pro_df_fin['Year_Month'] = pro_df_fin['Year'].astype(str)+" "+pro_df_fin['Month'].astype(str)


    pro_fin_dic[str(pro_nam_fin)] = pro_df_fin

filenames = []
#--------------------------------------------- UI----------------

# print(df_pro_info)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([

    html.H1("Project Dashboard"),

    dcc.Dropdown(
                id='pro_num',
                options = [{'label': i, 'value': i} for i in df_pro_info['Number']],
                value = df_pro_info['Number'].iloc[0]
                ,style={'width': '50%'}),

    html.Div(children=[
                html.Div(children=[
                    html.H6('Project_Name'),
                    html.H6(id='pro_nam'),
                    html.H6('Project_Manager'),
                    html.H6(id='pro_man')
                ], style={'width': '20%', 'display': 'inline-block', 'border': 'none', 'text-align': 'left'}),

            ], style={'width': '30%', 'vertical-align' : 'top', 'font-family' : 'Century Gothic', }),

    html.Div(children=[

        html.H4(children='Project Shedule Progress'),
        dcc.Graph(
            id='progress_data'
        )
    ]),

    html.Div(children=[

        html.H4(children='Project Finance Progress'),
        dcc.Graph(
            id='fin_data'
        )
    ])

 ])

@app.callback(
    [
     Output('pro_nam','children'),
     Output('pro_man','children'),
     Output('progress_data', 'figure')
    ],
    Input('pro_num', 'value')
    )

def update_pro_details(pro_num_sel):

    pro_num_sel = str(pro_num_sel)

    #--------------finding the project number in the project information------------

    for i in range(0,len(df_pro_info['Number']),1):
        if str(df_pro_info['Number'][i])==pro_num_sel:
            index = i
        else:
            pass
    #------------------------------------------------------
    fig = px.line(x=pro_dat_ext, y=[df_pla[pro_num_sel],df_act[pro_num_sel]], title="Project Progress")

    fig.update_layout(transition_duration=500, showlegend=True,
                      yaxis_title='% Progress',xaxis_title='Date Time')

    #-------------------this function for legend change---------

    def customLegend(fig, nameSwap):
        for i, dat in enumerate(fig.data):
            for elem in dat:
                if elem == 'name':
                    fig.data[i].name = nameSwap[fig.data[i].name]
        return(fig)

    fig = customLegend(fig=fig, nameSwap = {'wide_variable_0': 'Planned', 'wide_variable_1':'Actual'})

    #-------------------legend change function over------------

    return [df_pro_info['Name'][index],df_pro_info['Manager'][index],fig]


#-------------------update the figure witht the financial data----------------------
@app.callback(
    Output('fin_data','figure'),
    Input('pro_num', 'value')
    )

def update_fin_details(pro_num_sel):

    pro_num_sel = str(pro_num_sel)
    df = pro_fin_dic[pro_num_sel]


    #x= df['Year_Month']
    #x = range(1,len(df['A_value_LKR_cum'])+1,1)
    #y = [df['P_value_LKR_cum'],df['A_value_LKR_cum']]
    #y = df['P_value_LKR_cum']

    fig = px.line(df,x='Year_Month',y=['P_value_LKR_cum','A_value_LKR_cum'], title="Financial Progress")
    fig.update_xaxes(type='category')
    fig.update_layout(transition_duration=500, showlegend=True,
                      yaxis_title='LKR Milion',xaxis_title='Year Month')
    return fig

if __name__ == '__main__':
    app.run_server(debug=False, port=8051)