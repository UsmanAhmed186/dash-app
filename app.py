import plotly.graph_objects as go
import plotly.express as px
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Output, Input, State
import dash
from dash import dash_table
import pandas as pd
import numpy as np

go.Figure()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

CHARTSTYLE =  {
    'marginTop':10,
    'marginBottom':10, 
    'marginLeft':10, 
    'marginRight':10,
}

SLIDERSTYLE = {
    "display": "grid", 
    "grid-template-columns": "50%", 
    'width': '50%',
    'marginTop':10,
    'marginBottom':10, 
    'marginLeft':10, 
    'marginRight':10,
}

GRAPHSTYLE= {
    "background":'#FFFFFF', 
    'marginTop':0, 
    'marginBottom':0, 
    'marginLeft':0,
    'marginRight':0,
}

MAINSTYLE = {
    "background":'#FFFFFF', 
    'marginTop':40, 
    'marginBottom':40, 
    'marginLeft':40,
    'marginRight':40
}
CONTENTSTYLE = {
    'margin-left': '5%',
    'margin-right': '5%',
    'padding': '10px 10p',
}

TEXTSTYLE = {
    'textAlign': 'center',
    'color': '#191970',
    'marginTop':5,
    'marginBottom':5,
    'marginLeft':5,
    'marginRight':5
}

TABLEHEADERSTYLE = {
    'font-weight': 'bold',
    'text-transform': 'capitalize',
    'backgroundColor': '#636EFA',
    'color': 'white',
    'textAlign': 'center',
}

TABLESTYLE = {
    'backgroundColor': '#FFFFFF',
    'color': 'black',
    'textAlign': 'center',
}

DROPDOWNSTYLE = {
    'width':'70%',
    'marginTop':5,
    'marginBottom':5,
    'marginLeft':5,
    'marginRight':5,
    'textAlign': 'center',
}

def get_content():
    table2 = html.Ul(
        [
        dcc.Markdown(children='Hello')

            ])
            

    dropdown1 = dcc.Dropdown(id='wanProvider_dropdown', value='Total', 
                                style = DROPDOWNSTYLE,
                    options=[{'label': provider, 'value': str(provider)} 
                                for provider in wan_df_forecast.columns.tolist()+['All providers', 'Total'] 
                                if provider!='label' and provider!='date'])

    dropdown2 = dcc.Dropdown(id='train_dropdown', value='Total',   
                                style = DROPDOWNSTYLE,
                    options=[{'label': train, 'value': str(train)} for train in train_df_forecast['time_series'].unique().tolist()+['All trains', 'Total']])

    dataTable1 = dash_table.DataTable(
                    id='tbl1',
                    style_header=TABLEHEADERSTYLE,
                    style_data=TABLESTYLE,                 

                    columns=[{"name": i, "id": i} for i in train_df_forecast_latest.columns])

    dataTable2 = dash_table.DataTable(
                    id='tbl2',
                    style_header=TABLEHEADERSTYLE,
                    style_data=TABLESTYLE,                 

                    columns=[{"name": i, "id": i} for i in wan_df_forecast_latest.columns if i != 'label'])

    table1 = html.Div(
        [
            html.H4('Forecast Usage for Next 8 days'),
            dataTable1,
            html.Div(id='dataUsage_table1'),
        ]
    )
    table2 = html.Div(
        [
            html.H4('Forecast Usage for Next 8 days'),
            dataTable2,
            html.Div(id='dataUsage_table2'),
        ]
    )
    content_first_row = dbc.Row(
        [
            dbc.Col(
                [
                    html.H4('Provider Network Usage'),
                    dropdown1,
                    dcc.Graph(id='dataUsage_chart', style=GRAPHSTYLE),
                ]
            ),
            dbc.Col(
                [
                    html.H4('Train Network Usage'),
                    dropdown2,
                    dcc.Graph(id='dataUsageTrain_chart', style=GRAPHSTYLE),

                ]
            ),
            dbc.Col(
                [
                    table1,
                ]
            ),
        ]
    )

    slider1 = html.Div(
        [
            dbc.Row([ 
                html.H4('Usage per category',
                            style = CHARTSTYLE),
                dcc.Slider(id='num_category_slider',
                            min=1, max=10, step=1, included=True,
                            value=5,
                            marks={n: str(n) for n in range(1,11)}),
                html.Div(id='slider1', style=SLIDERSTYLE),
            ])
        ]
    )

    slider2= html.Div(
        [        
            dbc.Row([ 
                html.H4('Total Data Usage by Provider:',  
                            style = CHARTSTYLE),
                dcc.Slider(id='num_provider_slider',
                        min=1, max=10, step=1, included=True,
                        value=5,
                        marks={n: str(n) for n in range(1,11)}),
                html.Div(id='slider2', style=SLIDERSTYLE),

            ])
        ]
    )


    content_second_row = dbc.Row(
        [
            dbc.Col([
                dbc.Row([
                    slider1,
                    dcc.Graph(id='dataUsage_pieChart', style=GRAPHSTYLE),
                ])
            ]),
            dbc.Col([
                dbc.Row([
                    slider2,
                    dcc.Graph(id='dataUsage_donutChart', style=GRAPHSTYLE),
                ])
            ]),
            dbc.Col([table2])

        ]
    )

    content = html.Div(
        [
            html.H2('Analytics Dashboard Data Usage', style=TEXTSTYLE),
            html.Hr(),
            content_first_row,
            content_second_row,
        ],
        style=CONTENTSTYLE
    )

    return content

@app.callback(Output('dataUsage_chart', 'figure'),
              Input('wanProvider_dropdown', 'value'))
def plot_dataUsage_wanProvider(provider):
    # provider_df = wan_df_forecast[['date','time_series','dataUsage','Label']].sort_values('date', ascending=True)
    if provider=='All providers':
        provider_df = wan_df_forecast
        cols = [c for c in provider_df.columns if c!='label']
        fig = px.line(provider_df, y=cols, color='label', width=600, height=350)

    elif provider=='Total':
        provider_df = pd.DataFrame(wan_df_forecast.drop(['label','date'],axis=1).values.sum(axis=1), columns=['dataUsage'])
        provider_df['date'] = wan_df_forecast['date']
        provider_df['label'] = wan_df_forecast['label']
        
        fig = px.line(provider_df, x ='date', y='dataUsage', color='label', width=600, height=350)

    else:
        provider_df = wan_df_forecast[['date', provider,'label']]
        provider_df["dataUsage"] = provider_df[provider].rolling(3).mean()
        fig = px.line(provider_df, x="date", y=['dataUsage'], color='label', width=600, height=350)
    return fig

@app.callback(Output('dataUsageTrain_chart', 'figure'),
              Input('train_dropdown', 'value'))
def plot_dataUsage_trainProvider(train):
    if train=='All trains':
        train_df = train_df_forecast.groupby(['date','time_series']).mean()
        train_df['new'] = train_df.index.to_numpy()
        train_df.reset_index(inplace=True, drop=True)
        train_df['date'] = train_df.new.apply(lambda x:x[0])
        train_df['time_series'] = train_df.new.apply(lambda x:x[1])
        train_df.drop('new', inplace=True, axis=1)
        train_df = train_df[['dataUsage','date','Label','time_series']].fillna(0)
        train_df['date'] = pd.to_datetime(train_df['date'])
        train_df['dataUsage'] = train_df['dataUsage'].values+train_df['Label']
        train_df["dataUsage"] = train_df['dataUsage'].rolling(3).mean()
        train_df.drop('Label', axis=1, inplace=True)
        train_df = train_df.pivot(index='date', columns=['time_series']).fillna(0)
        train_df.columns = [c[1] for c in train_df.columns]
        
        fig = px.line(train_df, y=train_df.columns, width=600, height=350)

    elif train=='Total':
        train_df = train_df_forecast.groupby(['date']).sum()
        train_df = train_df[['dataUsage', 'Label']]
        fig = px.line(train_df, y=['dataUsage', 'Label'], width=600, height=350)

    else:
        train_df = train_df_forecast[['date','time_series','dataUsage','Label']].sort_values('date', ascending=True)
        train_df = train_df[train_df['time_series']==int(train)]
        train_df["dataUsage"] = train_df['dataUsage'].rolling(3).mean()
        fig = px.line(train_df, x="date", y=['dataUsage', 'Label'], width=600, height=350)
    return fig

@app.callback(Output('dataUsage_pieChart', 'figure'),
                Input('num_category_slider', 'value'))
def plot_pieChart(numOfDisplayedCats):
    category_df = df.drop(['monthOfYear', 'date'], axis=1)
    cols = category_df.columns.values
    category_df = pd.DataFrame({'category':cols,'dataUsage':category_df.sum(axis=0).values}).sort_values('dataUsage', ascending=False)
    fig = px.pie(category_df, values=category_df['dataUsage'][:numOfDisplayedCats].values, 
                              names=category_df['category'][:numOfDisplayedCats].values, 
                              width=600, height=400)

    return fig

@app.callback(Output('dataUsage_donutChart', 'figure'),
                Input('num_provider_slider', 'value'))
def plot_donutChart(numOfDisplayedProvider):
    wanProvider_df = wan_df.drop(['date'], axis=1)
    cols = wanProvider_df.columns.values
    wanProvider_df = pd.DataFrame({'wanProvider':cols,'dataUsage':wanProvider_df.sum(axis=0).values}).sort_values('dataUsage', ascending=False)
    fig = px.pie(wanProvider_df, values=wanProvider_df['dataUsage'][:numOfDisplayedProvider].values, 
                              names=wanProvider_df['wanProvider'][:numOfDisplayedProvider].values, 
                              width=600, height=400, hole=0.5)

    return fig

@app.callback(Output('tbl1', 'data'), 
            Input('train_dropdown', 'value'))
def update_table1(train):
    if train == 'All trains' or train == 'Total':
        train='5023'
    train_df = train_df_forecast_latest[train_df_forecast_latest['Train']==int(train)]
    train_df = train_df.sort_values('date', ascending=False)    
    train_df['Usage'] = train_df['Usage'].apply(lambda x: round(x,2))
    train_df['date'] = train_df['date'].dt.strftime('%d %B, %Y')
    data = train_df.to_dict('rows')
    return data

@app.callback(Output('tbl2', 'data'), 
            Input('wanProvider_dropdown', 'value'))
            
def update_table2(provider):
    if provider == 'All providers' or provider == 'Total':
        provider='TDC Denmark'
    provider_df = wan_df_forecast_latest
    cols = [c for c in provider_df.columns if c !='date' and c!= 'label']
    provider_df = provider_df.sort_values('date', ascending=False) 
    for c in cols: 
        provider_df[c] = provider_df[c].apply(lambda x: round(x,2))
    provider_df['date'] = provider_df['date'].dt.strftime('%d %B, %Y')
    data = provider_df.to_dict('rows')
    return data

if __name__ == '__main__':
    df = pd.read_csv('data_files/paloalto_daily_random_data.csv', index_col=0)
    wan_df= pd.read_csv('data_files/wan_df_final.csv')

    wan_df_forecast= pd.read_csv('data_files/wan_df_forecast.csv')
    wan_df_forecast['date'] = pd.to_datetime(wan_df_forecast['date'])
    wan_df_forecast_latest = wan_df_forecast[wan_df_forecast['label']=='prediction']
    # wan_df_forecast_latest = wan_df_forecast_latest.rename(columns={'Label':'Usage', 'time_series':'Provider'})
    # wan_df_forecast_latest = wan_df_forecast_latest[['date','Provider','Usage']]

    train_df_forecast= pd.read_csv('data_files/train_df_forecast.csv')
    train_df_forecast['date'] = pd.to_datetime(train_df_forecast['date'])
    train_df_forecast_latest = train_df_forecast[train_df_forecast['Label'].notna()]
    train_df_forecast_latest = train_df_forecast_latest.rename(columns={'Label':'Usage', 'time_series':'Train'})
    train_df_forecast_latest = train_df_forecast_latest[['date','Train','Usage']]

    app.layout  = get_content()

    app.run_server(port=8051, debug=True)