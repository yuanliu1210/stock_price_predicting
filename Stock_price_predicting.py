#import libraries
import requests
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas_datareader.data as web
import datetime
from bs4 import BeautifulSoup
from IPython.core.display import HTML
import time
from dash.dependencies import Output,  Event, Input
import plotly.graph_objs as go
import base64
import math
import pandas as pd 
import numpy as np
from sklearn.svm import SVR
import sqlite3
import copy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#******************************************************************************************************************************************************************************************************************************************************************************************************************
tt = datetime.datetime.now()
stock = 'TSLA'
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#******************************************************************************************************************************************************************************************************************************************************************************************************************
#time variables
# current time
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
now_stamp = time.time()

# today's open & close time
todaydate = now[0:10]
opentime = todaydate + ' 09:30:00'
closetime = todaydate + ' 16:30:00'

# today's open & close timestamp
open_stamp = time.mktime(datetime.datetime.strptime(opentime, "%Y-%m-%d %H:%M:%S").timetuple())
close_stamp = time.mktime(datetime.datetime.strptime(closetime, "%Y-%m-%d %H:%M:%S").timetuple())

# yesterday open & close time / timestamp
now_obj = datetime.datetime.now()
yesterday_open = datetime.datetime(now_obj.year,now_obj.month,now_obj.day-1,9,30,0)
yesterday_close = datetime.datetime(now_obj.year,now_obj.month,now_obj.day-1,16,30,0)
open_stamp_y = time.mktime(yesterday_open.timetuple())
close_stamp_y = time.mktime(yesterday_close.timetuple())

#******************************************************************************************************************************************************************************************************************************************************************************************************************
#different senarios for live-update graph
#live = True
day = datetime.datetime.today().weekday()
# if today is Saturday
if day == 5:
    stamp1 = open_stamp_y
    stamp2 = close_stamp_y
    live = False
    fdate = datetime.datetime.fromtimestamp(open_stamp_y).strftime("%Y-%m-%d %H:%M:%S")
    stockdate = fdate[0:10]

# if today is Sunday
elif day == 6:
    # friday's open & close time / timestamp
    friday_open = datetime.datetime(now_obj.year,now_obj.month,now_obj.day-2,9,30,0)
    friday_close = datetime.datetime(now_obj.year,now_obj.month,now_obj.day-2,16,30,0)
    open_stamp_f = time.mktime(friday_open.timetuple())
    close_stamp_f = time.mktime(friday_close.timetuple())
    stamp1 = open_stamp_f
    stamp2 = close_stamp_f
    live = False
    fdate = datetime.datetime.fromtimestamp(open_stamp_f).strftime("%Y-%m-%d %H:%M:%S")
    stockdate = fdate[0:10]
    
elif day == 0:
    if now_stamp < open_stamp:
        friday_open = datetime.datetime(now_obj.year,now_obj.month,now_obj.day-3,9,30,0)
        friday_close = datetime.datetime(now_obj.year,now_obj.month,now_obj.day-3,16,30,0)
        open_stamp_f = time.mktime(friday_open.timetuple())
        close_stamp_f = time.mktime(friday_close.timetuple())
        stamp1 = open_stamp_f
        stamp2 = close_stamp_f
        live = False
        fdate = datetime.datetime.fromtimestamp(open_stamp_f).strftime("%Y-%m-%d %H:%M:%S")
        stockdate = fdate[0:10]
    elif now_stamp > close_stamp:
        #whole day
        stamp1 = open_stamp
        stamp2 = close_stamp
        live = False
        stockdate = todaydate
    else:
        stamp1 = open_stamp
        stamp2 = now_stamp
        live = True
        stockdate = todaydate
else:
    if now_stamp < open_stamp:
        #yesterday
        stamp1 = open_stamp_y
        stamp2 = close_stamp_y
        live = False
        ydate = datetime.datetime.fromtimestamp(open_stamp_y).strftime("%Y-%m-%d %H:%M:%S")
        stockdate = ydate[0:10]
    elif now_stamp > close_stamp:
        #whole day
        stamp1 = open_stamp
        stamp2 = close_stamp
        live = False
        stockdate = todaydate
    else:
        #up-to-now
        stamp1 = open_stamp
        stamp2 = now_stamp
        stockdate = todaydate
        live = True

#********************************************************************************************************************************************
#Web scraping for stock prices

#get data of stock price from yahoo website
url1 = 'https://query1.finance.yahoo.com/v8/finance/chart/TSLA?symbol=TSLA&period1='
url2 = '&period2='
url3 = '&interval=5m&includePrePost=true&events=div%7Csplit%7Cearn&lang=en-US&region=US&crumb=mfU1rq6PZgP&corsDomain=finance.yahoo.com'
stam1 = str(int(stamp1))
stam2 = str(int(stamp2))
url = url1 + stam1 + url2 + stam2 + url3
print (stam1, stam2)
response = requests.get(url)
print(response.status_code, '\n')
js = response.json()

#get timestamp and stockprice from historical data
timestamp = js['chart']['result'][0]['timestamp']
stockprice = js['chart']['result'][0]['indicators']['quote'][0]['close']

#******************************************************************************************************************************************************************************************************************************************************************************************************************
#Store Price in SQL database

list1 = zip(timestamp, stockprice)
conn = sqlite3.connect('TSLA.db')
c = conn.cursor()
try:
        
    c.executescript('DROP TABLE if exists my_sql')

    c.execute("CREATE TABLE my_sql (timestamp INT, stockprice FLOAT)")                                                                                    
    c.executemany("INSERT INTO my_sql VALUES(?, ?)", list1)
    conn.commit()
    
except:
    pass

# Create your connection.
tsla = sqlite3.connect('TSLA.db')

dfsql = pd.read_sql_query("SELECT * FROM my_sql", tsla)
timestamp1 = dfsql.timestamp.values
stockprice1 = dfsql.stockprice.values

conn.close()

X = []
Y = stockprice1
Y = Y.tolist()
for ts in timestamp1:
    td = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    t = td[11:19]
    X.append(t)

#******************************************************************************************************************************************************************************************************************************************************************************************************************
#News Scraping

r = requests.get('https://ca.finance.yahoo.com/quote/TSLA/news?p=TSLA')
soup1 = BeautifulSoup(r.text,'lxml')

#get the list of news url
list_of_links = []
for li in soup1.find_all('li'):
    try:
        list_of_links.append(li.find('a').get('href'))
    except:
        pass
list_of_links = list_of_links[10:-7]

base_news_url = 'https://ca.finance.yahoo.com'
news_url = []                    
for i in range(len(list_of_links)):
    news_url.append(base_news_url + list_of_links[i])

#Get news title, content for each news url
news = {}
for url in news_url:
    try:
        r = requests.get(url)
        news_soup = BeautifulSoup(r.text, 'lxml') 
    except:
        pass 
    
    header = news_soup.header.text
    article = [] 
    article.append(header)
    contents = ''
    
    for p in news_soup.find_all('p'):
        try: 
            contents += re.sub('<.*?>' , '' , p.get('content'))
        except:
            pass

    article.append(contents) 
    news[url] = article
    
#create copy of original(unparsed) news article content for dashboard display
origin_news = copy.deepcopy(news)

#******************************************************************************************************************************************************************************************************************************************************************************************************************
#NLP Analysis

ps = PorterStemmer()
stop_words =  set(stopwords.words('english')+list(punctuation))

def tokenized(text):
    words = word_tokenize(text)
    # stop words - delete common word and punctuation
    filtered_sentence = []
    for w in words:
        if w not in stop_words:
            filtered_sentence.append(w.lower())
    return filtered_sentence

# stemming
def stem(filtered_sentence):
    stemmed_list = []
    for n in filtered_sentence:
        stemmed_word = ps.stem(n)
        stemmed_list.append(stemmed_word)
    return stemmed_list  
    
for header in news.keys():
    content = news[header]
    content[1] = tokenized(content[1])
    content[1] = stem(content[1])

#******************************************************************************************************************************************************************************************************************************************************************************************************************
#News WordCloud  

contents = []
for header in news.keys():
    content = news[header]
    for word in content[1]:
        contents.append(word)
unique_string=(" ").join(contents)

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords = stop_words)
wordcloud = wordcloud.generate(unique_string)
fig1 = plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.title("WordCloud of Daily News")
plt.axis("off")
fig1.savefig("tesla.png")
#******************************************************************************************************************************************************************************************************************************************************************************************************************
#Calculate IDF 

#count total word-tesla and document size
countTesla = 0
for header in news.keys():
    content = news[header]
    for i in content[1]:
        if i == 'tesla':
            countTesla+=1

total_document = len(news)

def each_term_freq(word, each_news_content):   
    return float(each_news_content.count(word))/(len(each_news_content)+1)

for each_news in news:
    content = news[each_news]
    tesla_tf =  each_term_freq('tesla', content[1])

# compute the idf for tesla in the vocabulary
document_contain_tesla = 0
a = 0
for each_news in news:
    content = news[each_news]
    for word in content[1]:
        if word == 'tesla':
            a = 1
    document_contain_tesla += a
               
tesla_idf = 1 + math.log(total_document/float(document_contain_tesla))

for each_news in news:
    title = news[each_news][0]
    article = origin_news[each_news][1]
    stem_content = news[each_news][1]
    tesla_tf =  each_term_freq('tesla', stem_content)
    tesla_tfidf = tesla_tf * tesla_idf
    origin_news[each_news].append(tesla_tfidf)

#******************************************************************************************************************************************************************************************************************************************************************************************************************
# Change News Articles into Pandas DataFrame

df = pd.DataFrame(data=origin_news, index=['title','content','TFIDF'])  
# sort news by tfidf scores descending
df = df.T.sort_values('TFIDF',ascending= False)
# select news have TFIDF greater than 0
df = df.loc[df['TFIDF'] >0]
# remove news with same title, keep the news with highest TFIDF score 
df = df.drop_duplicates(subset='title',keep='first')

#******************************************************************************************************************************************************************************************************************************************************************************************************************
# News Sentiment Analysis

objects = []
performance = []
overall_score =[]
analyzer =  SentimentIntensityAnalyzer()
negative = 0
neural = 0
positive = 0

# compound score of top 5 news articles
for i in range(min(5, len(df))):
    content = (df.iloc[i]['content'])
    vs = analyzer.polarity_scores(content)
    negative += vs['neg']
    neural += vs['neu']
    positive += vs['pos']
    performance.append(vs['compound'])
    objects.append(str('news')+str(i))

fig2 = plt.figure()
y_pos = np.arange(len(objects))
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Compound Score')
plt.title( 'Sentiment Score of News')
fig2.savefig("sentiment.png")

#******************************************************************************************************************************************************************************************************************************************************************************************************************
#generate data table for dash with hyperlinks

def Table(dataframe):
    rows = []
    for i in range(min(5, len(dataframe))):
        for col in dataframe.columns:

            value = dataframe.iloc[i][col]

            if col == 'title':
                title = html.Td(html.A(href= dataframe.iloc[i].name ,target="_blank",children =value))
                rows.append(html.Tr(title))
        
            elif col == 'content':
                sentences = sent_tokenize(value)
                content = " ".join(sentences[:1])
                content_displayed = html.Td(children = content)
                rows.append(html.Tr(content_displayed))

    return html.Table(
        rows
    )
#******************************************************************************************************************************************************************************************************************************************************************************************************************
#predict price

def predict_price (date,  price, x):
    date = np.reshape(date, (len(date),  1))
    svr_lin = SVR(kernel = 'linear',  C = 1e3)
    svr_ploy = SVR(kernel = 'poly',  C = 1e3,  degree = 2,  gamma = 'auto')
    svr_rbf = SVR(kernel = 'rbf',  C = 1e3,  gamma = 0.1)
    svr_lin.fit(date,  price)
    svr_ploy.fit(date,  price)
    svr_rbf.fit(date,  price)
    tlin = svr_lin.predict(date)
    tploy = svr_ploy.predict(date)
    trbf = svr_rbf.predict(date)
    plin = svr_lin.predict(x)
    pploy = svr_ploy.predict(x)
    prbf = svr_rbf.predict(x)
    return tlin, tploy,  trbf, plin,  pploy,  prbf

start = tt - datetime.timedelta(days=30)
end = tt
dfp = web.DataReader(stock, 'iex',  start,  end)
train_price = dfp.close.values #train-y

n = len(train_price)
pastdays = list(range(0, n)) #past dates
nx = list(range(n,n+5)) #next 5 days
x1 = list(range(0, len(nx)))
nx = np.reshape(nx, (len(nx), 1))

predicted = predict_price(pastdays, train_price, nx)

#******************************************************************************************************************************************************************************************************************************************************************************************************************
#Dashboard using Dash

#dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#wordcloud image
image_filename1 = 'tesla.png' 
encoded_image1   = base64.b64encode(open(image_filename1, 'rb').read())
#sentiment analysis image
image_filename2 = 'sentiment.png' 
encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())


#tabs styles
tabs_styles = {
    'height': '44px'}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'}
tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'}

#dash app layout
app.layout = html.Div([
    #header with image
    html.Div([
        html.H1('Finance Explorer -- TESLA ',
                style={'display': 'inline',
                       'font-size': '2.65em',
                       'margin-left': '7px',
                       'font-weight': 'bolder',
                       'font-family': 'Product Sans',
                       'color': "rgba(193, 66, 66, 0.95)", #117, 117, 117, 0.95
                       'margin-top': '20px',
                       'margin-bottom': '0'
                       }),
        html.Img(src="https://s3.amazonaws.com/techdecisions/wp-content/uploads/2018/05/22155507/Tesla-Motors-symbol.png",
                style={
                    'margin-top': '5px',
                    'height': '50px',
                    'float': 'right'
                })
        ], 
        style = {
            'borderBottom': 'thin lightgrey solid', 
            'backgroundColor': 'rgb(250,250,250)', 
            'padding': '10px 5px'
        }
    ),
    
    #create Dash tabs 
    dcc.Tabs(id="tabs", children=[
        #tab1 of stock price
        dcc.Tab(label='Stock Price', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div([
                 dcc.Interval(
                                    id='interval-component',
                                    interval=30*1000, # in milliseconds
                                    n_intervals=0), 
                html.Div([
                    html.Div(id='live-update-text',
                                    style={'font-size': '25px', 'width': '48%', 'display': 'inline-block',  }
                        ), 
                    #dropdown / radio items
                    html.Div([
                        dcc.Dropdown(
                            id='yaxis-indicators',
                            options=[{'label': i, 'value': i} for i in ['Close Price',  'Moving Average (50 days)',  'High/Low',  'Volume']],
                            value='Close Price'
                        ),
                        dcc.RadioItems(
                            id='xaxis-time',
                            options=[{'label': i, 'value': i} for i in ['1 month', '6 months',  '1 year',  '3 years']],
                            value='1 month',
                            labelStyle={'display': 'inline-block'}
                        )], 
                        style={'width': '48%', 'display': 'inline-block'}
                    )], 
                    style = {
                        'backgroundColor': 'rgb(250,250,250)', 
                        'padding': '10px 5px'
                    }),     
        
                 html.Div([
                    #live update
                     html.Div([
                            dcc.Graph(id='live-graph'), 
                            dcc.Interval(
                            id='graph-update', 
                            interval=30*1000
                            )], 
                            style={'width': '48%', 'display': 'inline-block'}
                    ), 
                    #history graph
                    html.Div([
                        dcc.Graph(id='historical graph')],
                        style={'width': '48%', 'display': 'inline-block'})
                ], 
                    style = {
                        'borderBottom': 'thin lightgrey solid', 
                        'backgroundColor': 'rgb(250,250,250)', 
                        'padding': '10px 5px'}
                )
            ])
        ]),  
        #tab2 of news
        dcc.Tab(label='News', style=tab_style, selected_style=tab_selected_style,  children=[
            html.Div([
                #sub titles
                html.Div([
                    html.Div(children='Daily News Headline', 
                        style={'width': '48%', 'display': 'inline-block', 'textAlign': 'center', 'fontSize': 24,  'fontFamily':'Courier New'}),
                    html.Div(children='News Sentiment Anaysis', 
                        style={'width': '48%', 'display': 'inline-block', 'textAlign': 'center',  'fontFamily':'Courier New',  'fontSize': 24}),
                ], 
                    style = {
                        'backgroundColor': 'rgb(250,250,250)', 
                        'padding': '10px 5px'}),     
                        
                html.Div([
                    #list of news
                    html.Div(
                        children = [ Table(df)], 
                        style={'width': '48%', 'display': 'inline-block'}, 
                        className="six columns"), 
                        
                    html.Div([
                        #wordcould image
                        html.Div([
                            html.Img(src='data:image/png;base64,{}'.format(encoded_image1.decode()),
                             style={'height': '49%', 'display': 'inline-block'})
                        ]),
                        #sentiment analysis image
                        html.Div([
                            html.Img(src='data:image/png;base64,{}'.format(encoded_image2.decode()),
                             style={'height': '49%', 'display': 'inline-block'})
                        ]),
                        
                    ],  className="six columns"),  
                ],
                )
                
            ])
        ]), 
        #tab3 of stock price prediction
        dcc.Tab(label='Prediction', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div([
                html.Div(children='Please select a predictive model', 
                        style={'width': '48%', 'display': 'inline-block',  'textAlign': 'center', 'fontSize': 16,  'fontFamily':'Courier New'}),
                html.Div([
                    dcc.Dropdown(
                    id='yaxis_model',
                    options=[{'label': i, 'value': i} for i in ['Linear SVR',  'Polynomial SVR',  'RBF SVR']],
                    value ='Linear SVR'
                )],
                style={'width': '48%', 'display': 'inline-block'})
            ], 
             style = {
                        'borderBottom': 'thin lightgrey solid', 
                        'backgroundColor': 'rgb(250,250,250)', 
                        'padding': '10px 5px'}), 
            
            #model graph, predict graph
            html.Div([
                html.Div([
                    dcc.Graph(id='model graph', 
                        figure = {
                            'data': [
                                {'x': pastdays ,  'y': predicted[0],  'type':'line',  'name': 'Linear Model'}, 
                                {'x': pastdays ,  'y': predicted[1],  'type':'line',  'name': 'Polynomial Model'}, 
                                {'x': pastdays ,  'y': predicted[2],  'type':'line',  'name': 'RBF Model'}, 
                                go.Scatter(x = pastdays ,  y = train_price,  name = 'Actual Price',  mode = 'markers',  marker={'color': 'black'})
                            ], 
                            'layout':{
                                        'title': 'Models based on past 30 days', 
                                        'xaxis': {'title': 'Time (past days)'}, 
                                        'yaxis': {'title': 'Stock Price'}
                            }
                        }
                    )
                ],
                    style={'width': '48%', 'display': 'inline-block'}), 
             html.Div([
                dcc.Graph(id='predict graph')],
                style={'width': '48%', 'display': 'inline-block'}), 
            ],
                 style = {
                            'borderBottom': 'thin lightgrey solid', 
                            'backgroundColor': 'rgb(250,250,250)', 
                            'padding': '10px 5px'})
        ])
    ],  style=tabs_styles)
])

#******************************************************************************************************************************************************************************************************************************************************************************************************************
#live-updated text
@app.callback(Output('live-update-text', 'children'),
          [Input('interval-component', 'n_intervals')])
def update_time(n):
    return [html.P('Last updated '+str(datetime.datetime.now()))]

#******************************************************************************************************************************************************************************************************************************************************************************************************************
#historical graph
@app.callback(
    dash.dependencies.Output('historical graph', 'figure'),
    [dash.dependencies.Input('yaxis-indicators', 'value'),
     dash.dependencies.Input('xaxis-time', 'value')])

def historical_graph(yaxis_indicators_name, xaxis_time_name):
    if xaxis_time_name == '1 month':
        start = tt - datetime.timedelta(days=30)
        end = tt
    elif xaxis_time_name == '6 months':
        start = tt - datetime.timedelta(days=6*30)
        end = tt
    elif xaxis_time_name == '1 year':
        start = tt - datetime.timedelta(days=365)
        end = tt
    elif xaxis_time_name == '3 years':
        start = tt - datetime.timedelta(days=3*365)
        end = tt
    df = web.DataReader(stock, 'iex',  start,  end)
    df['MA'] = df.close.rolling(50).mean()
    
    if yaxis_indicators_name == 'Close Price': 
       close = {'x': df.index ,  'y': df.close,  'type':'line',  'name': 'Stock Price'}
       data = [close]
    elif yaxis_indicators_name == 'Moving Average (50 days)':
        close = {'x': df.index ,  'y': df.close,  'type':'line',  'name': 'Stock Price'}
        ma = go.Scatter(x = df.index ,  y = df.MA,   name= 'MA',  line = dict(color='rgb(204,153,255)'))
        data = [close, ma]
    elif yaxis_indicators_name == 'High/Low':
        close = {'x': df.index ,  'y': df.close,  'type':'line',  'name': 'Stock Price'}
        high = go.Scatter(x = df.index ,  y = df.high,  name = 'High',  line = dict(color='rgb(0,204,0)', dash = 'dot'))
        low = go.Scatter(x = df.index ,  y = df.low,  name = 'Low',  line = dict(color='rgb(255,51,51)',  dash = 'dot'))
        data = [close, high, low]
    elif yaxis_indicators_name == 'Volume':
        close = {'x': df.index ,  'y': df.close,  'type':'line',  'name': 'Stock Price'}
        volume = go.Scatter(x = df.index ,  y = df.volume,  fill = 'tonexty',    name = 'Volume',  yaxis = 'y2',  line = dict(color='rgb(255,178,102)'))
        data = [close, volume]
    return {
        'data': data,
        'layout':  go.Layout(
            title = 'Historical Stock Price of TSLA', 
            xaxis = dict(title = 'Time'), 
            yaxis = dict(title='Stock Price'), 
            yaxis2 = dict(title='Volume',  titlefont=dict(color='rgb(148,103,189)'),  
            tickfont=dict(color = 'rgb(148,103,189)'),  overlaying = 'y',  side = 'right')
        )
    }
    
#******************************************************************************************************************************************************************************************************************************************************************************************************************
#live-updated graph
@app.callback(Output('live-graph',  'figure'),  #will not work until pass figure into it
                        events = [Event('graph-update',  'interval')]) 
def update_graph():
    global X
    global Y
    if live:
        response = requests.get("https://finance.yahoo.com/quote/TSLA?p=TSLA")
        content = response.content
        soup = BeautifulSoup(content,  'html.parser')
        pr = soup.find ("span",  attrs = {"data-reactid":'14'})
        currentprice = pr.text
        #currentprice = currentp.tolist()
        currenttime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ct = currenttime[11:19]
        
        X.append(ct)
        Y.append(float(currentprice))
        
        data = go.Scatter(
            x = list(X), 
            y = list(Y), 
            name = 'Daily Stock Price of TSLA on ' + stockdate, 
            mode = 'lines+markers'
            )
    else:
        data = go.Scatter(
            x = list(X), 
            y = list(Y), 
            name = 'Daily Stock Price of TSLA on ' + stockdate, 
            
        )
        
    return {'data':[data],  
                'layout':  go.Layout(
                        title = 'Live-updated Daily Stock Price on ' + stockdate, 
                        xaxis = dict(title = 'Time', range=[min(X),  max(X)]), 
                        yaxis = dict(title = 'Stock Price', range=[min(Y),  max(Y)]))}
#******************************************************************************************************************************************************************************************************************************************************************************************************************
#predict graph
@app.callback(
    dash.dependencies.Output('predict graph', 'figure'),
    [dash.dependencies.Input('yaxis_model', 'value')])
    
def predictgraph(yaxis_model):
    if yaxis_model == 'Linear SVR': 
        data= {'x':  x1,  'y': predicted[3],  'type':'line',  'name': 'Linear SVR'}
    elif yaxis_model  == 'Polynomial SVR':
        data= {'x': x1 ,  'y': predicted[4],  'type':'line',  'name': 'Polynomial SVR'}
    elif yaxis_model == 'RBF SVR':
        data= {'x': x1,  'y': predicted[5],  'type':'line',  'name': 'RBF SVR'}
    return {
        'data': [data],
        'layout':  go.Layout(
            title = 'Next 5 days predicted prices', 
            xaxis = dict(title = 'Time'), 
            yaxis = dict(title='Stock Price')
        )
    }
#******************************************************************************************************************************************************************************************************************************************************************************************************************
if __name__ == '__main__':
    app.run_server()
