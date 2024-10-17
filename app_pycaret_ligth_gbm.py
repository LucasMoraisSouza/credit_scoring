# Imports
import pandas as pd
import numpy as np
import streamlit as st
import os

from io import BytesIO
from pycaret.classification import load_model, predict_model

# Função para converter o DataFrame para CSV
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Função para converter o DataFrame para Excel
@st.cache_data
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer._save()  # Atualizado para evitar problemas de depreciação
    processed_data = output.getvalue()
    return processed_data

# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação - Precisa ser a primeira chamada de Streamlit
    st.set_page_config(page_title='PyCaret', layout="wide", initial_sidebar_state='expanded')

    # Título principal da aplicação
    st.write("## Escorando o modelo gerado no PyCaret")
    st.write("---")
    st.write("##### Na coluna ao lado você verá um botão onde poderá fazer upload da sua base para que possamos fazer as previsões automaticamente!")
    st.write("##### Ao realizar o upload, basta aguardar o botão de dowload aparecer logo abaixo na tela.")
    st.markdown("---")
    
    # Botão para carregar arquivo na aplicação
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Bank Credit Dataset", type=['csv', 'ftr'])

    
    if data_file_1 is not None:

        # Carregar arquivo .feather
        df_credit = pd.read_feather(data_file_1)
        df_credit = df_credit.sample(50000)

        model_saved = load_model('model_final_lgbm_17_10_2024')
        predict = predict_model(model_saved, data=df_credit)

        # Converter previsões para Excel
        df_xlsx = to_excel(predict)

        # Botão para baixar o arquivo
        st.download_button(label='📥 Download', data=df_xlsx, file_name='predict.xlsx')

        st.write("##### Pronto! Agora basta baixar o arquivo excel clicando no botão acima para visualizar todas previsões realizadas na base de dados que você escolheu.")
        st.write("---")

if __name__ == '__main__':
    main()