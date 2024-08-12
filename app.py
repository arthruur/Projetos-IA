import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Dados e processamento
colunas_relevantes = [
    'TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_ESCOLA',
    'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT',
    'CO_MUNICIPIO_PROVA', 'NO_MUNICIPIO_PROVA', 'CO_UF_PROVA', 'SG_UF_PROVA'
]

microdadosEnem = pd.read_csv(
    "C:/Users/Arthur/Downloads/microdados_enem_2019/DADOS/MICRODADOS_ENEM_2019.csv",
    sep=";", 
    encoding='ISO-8859-1',
    usecols=colunas_relevantes
)

faixas_etarias = {
    1: 'Menor de 17 anos',
    2: '17 anos',
    3: '18 anos',
    4: '19 anos',
    5: '20 anos',
    6: '21 anos',
    7: '22 anos',
    8: '23 anos',
    9: '24 anos',
    10: '25 anos',
    11: 'Entre 26 e 30 anos',
    12: 'Entre 31 e 35 anos',
    13: 'Entre 36 e 40 anos',
    14: 'Entre 41 e 45 anos',
    15: 'Entre 46 e 50 anos',
    16: 'Entre 51 e 55 anos',
    17: 'Entre 56 e 60 anos',
    18: 'Entre 61 e 65 anos',
    19: 'Entre 66 e 70 anos',
    20: 'Maior de 70 anos'
}
microdadosEnem['Faixa Etária'] = microdadosEnem['TP_FAIXA_ETARIA'].map(faixas_etarias)

# Inicializar o app
app = dash.Dash(__name__)


# Layout do Dashboard
app.layout = html.Div([
    html.H1("Dashboard Interativo - ENEM 2019"),

    dcc.Dropdown(
        id='municipio-dropdown',
        options=[{'label': municipio, 'value': municipio} for municipio in sorted(microdadosEnem['NO_MUNICIPIO_PROVA'].dropna().unique())],
        value='Feira de Santana',  # Valor padrão
        clearable=False
    ),

    # Adicionar o total de participantes aqui
    html.Div(id='total-participantes', style={'fontSize': 24, 'marginTop': 20}),

    dcc.Graph(id='distribuicao-etaria-grafico'),
    dcc.Graph(id='proporcao-genero-grafico'),
    dcc.Graph(id='desempenho-comparativo-grafico'),
    dcc.Graph(id='desempenho-medio-grafico')
])

# Callbacks para atualizar os gráficos
@app.callback(
    Output('distribuicao-etaria-grafico', 'figure'),
    Input('municipio-dropdown', 'value'))
def update_etaria_grafico(municipio):
    if municipio == 'Nacional':
        distribuicao = microdadosEnem['Faixa Etária'].value_counts().sort_index()
    else:
        cidade = microdadosEnem[microdadosEnem['NO_MUNICIPIO_PROVA'] == municipio]
        distribuicao = cidade['Faixa Etária'].value_counts().sort_index()

    fig = px.bar(distribuicao, x=distribuicao.index, y=distribuicao.values,
                 title=f'Distribuição Etária - {municipio}', labels={'x': 'Faixa Etária', 'y': 'Número de Participantes'})
    return fig

@app.callback(
    Output('proporcao-genero-grafico', 'figure'),
    Input('municipio-dropdown', 'value'))
def update_genero_grafico(municipio):
    if municipio == 'Nacional':
        proporcao = microdadosEnem['TP_SEXO'].value_counts(normalize=True) * 100
    else:
        cidade = microdadosEnem[microdadosEnem['NO_MUNICIPIO_PROVA'] == municipio]
        proporcao = cidade['TP_SEXO'].value_counts(normalize=True) * 100

    proporcao.index = proporcao.index.map({'M': 'Masculino', 'F': 'Feminino'})
    fig = px.bar(proporcao, x=proporcao.index, y=proporcao.values,
                 title=f'Proporção de Candidatos por Gênero - {municipio}', labels={'x': 'Gênero', 'y': 'Proporção (%)'})
    return fig

@app.callback(
    Output('desempenho-comparativo-grafico', 'figure'),
    Input('municipio-dropdown', 'value'))
def update_comparativo_grafico(municipio):
    escolas_publicas = microdadosEnem[microdadosEnem['TP_ESCOLA'] == 2]
    escolas_privadas = microdadosEnem[microdadosEnem['TP_ESCOLA'] == 3]

    desempenho_publico = escolas_publicas[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT']].describe().loc['mean']
    desempenho_privado = escolas_privadas[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT']].describe().loc['mean']

    fig = px.bar(x=desempenho_publico.index, y=desempenho_publico.values, labels={'x': 'Área do Conhecimento', 'y': 'Nota Média'},
                 title='Desempenho Médio em Escolas Públicas e Privadas')
    fig.add_bar(x=desempenho_privado.index, y=desempenho_privado.values, name='Escolas Privadas')
    return fig

@app.callback(
    Output('desempenho-medio-grafico', 'figure'),
    Input('municipio-dropdown', 'value'))
def update_desempenho_medio_grafico(municipio):
    notas = microdadosEnem[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT']].dropna()
    media_notas = notas.mean()

    fig = px.bar(media_notas, x=media_notas.index, y=media_notas.values, labels={'x': 'Área do Conhecimento', 'y': 'Nota Média'},
                 title='Desempenho Médio por Área do Conhecimento')
    return fig

@app.callback(
    Output('total-participantes', 'children'),
    Input('municipio-dropdown', 'value')
)
def atualizar_total(municipio_selecionado):
    # Filtrar os dados com base no município selecionado
    total_participantes = microdadosEnem[microdadosEnem['NO_MUNICIPIO_PROVA'] == municipio_selecionado].shape[0]
    
    # Retornar o total como um texto
    return f"Total de Participantes: {total_participantes}"


# Executar o app
if __name__ == '__main__':
    app.run_server(debug=True)
