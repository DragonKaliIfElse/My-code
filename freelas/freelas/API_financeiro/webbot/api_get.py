import requests
import base64
import json
import re
from typing import Dict, List, Any

class SiengeAPI:
    def __init__(self, subdomain: str, password: str, token: str = '', start_date: str = '', end_date: str = '', customer_id: int = 1):
        self.subdomain = subdomain
        self.password = password
        self.token = token or self._generate_token()
        self.base_url = f"https://api.sienge.com.br/{subdomain}/public/api/v1"
        self.bulk_data_url = f"https://api.sienge.com.br/{subdomain}/public/api/bulk-data/v1"
        self.start_date = start_date
        self.end_date = end_date
        self.customer_id = customer_id

        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Basic {self.token}"
        }
 
    def _generate_token(self):
        credentials = f"{self.subdomain}:{self.password}"
        return base64.b64encode(credentials.encode()).decode()

    def set_start_date(self, start_date):
        self.start_date = start_date

    def set_end_date(self, end_date):
        self.end_date = end_date

    def set_customer_id(self, customer_id):
        self.customer_id = customer_id

    def get_enterprises(self, offset, limit):
        resource = f"enterprises?offset={offset}&limit={limit}"
        url = f"{self.base_url}/{resource}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def get_contracts(self, offset, limit):
        resource = f"supply-contracts/all?contractStartDate={self.start_date}&contractEndDate={self.end_date}&limit={limit}&offset={offset}"
        url = f"{self.base_url}/{resource}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def get_receivable_bills(self, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
        resource = f"accounts-receivable/receivable-bills?customerId={self.customer_id}&startDueDate={self.start_date}&endDueDate={self.end_date}&page={page}&pageSize={page_size}"
        url = f"{self.base_url}/{resource}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()
 
    def get_bills(self, limit, offset):
        resource = f"bills/?startDate={self.start_date}&endDate={self.end_date}&limit={limit}&offset={offset}"
        url = f"{self.base_url}/{resource}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def get_account_statement(self, limit, offset):
        resource = f"accounts-statements?startDate={self.start_date}&endDate={self.end_date}&offset={offset}&limit={limit}"
        url = f"{self.base_url}/{resource}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def get_cost_centers(self, limit, offset): 
        resource = f"cost-centers?offset={offset}&limit={limit}"
        url = f"{self.base_url}/{resource}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def get_cost_estimation_itens(self, buildingId):
        resource = f"building-cost-estimation-items?buildingId={buildingId}"
        url = f"{self.bulk_data_url}/{resource}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

def all_data(func):
    limit = 200
    empresas = func(offset=0, limit=limit)
    count = empresas['resultSetMetadata']['count']
    while limit < count:
        emp2 = func(offset=limit, limit=200)
        empresas['results'].extend(emp2['results'])
        limit += 200
    return empresas

# Exemplo de uso
def main():
    # Inicializa a API
    api = SiengeAPI(
        subdomain="engeplan",
        password="l2dCJQaD5y21QB009C8J7nEawnpt1jmR",
        token="ZW5nZXBsYW4tYmk6bDJkQ0pRYUQ1eTIxUUIwMDlDOEo3bkVhd25wdDFqbVI",  # ou deixa None para gerar automaticamente
        start_date="2025-09-01",
        end_date="2025-10-01"
    )

    try:
        dados_contratos = all_data(api.get_contracts)

        #bills (dividas e faturas) mensais
        #dividas = sum([ fat['totalInvoiceAmount'] for fat in empresas['results'] ])
        #fatura = sum([ fat['totalInvoiceAmount'] if fat["installmentsNumber"] == 1 else fat['totalInvoiceAmount']/fat['installmentsNumber'] for fat in empresas['results'] ])
        #print(f'total de faturas no mês de setembro: {fatura}')
        #print(f'valor total a pagar: {dividas}')

        #accounts-statements (extrato)
        #total_pagamento = sum([ extrato['value'] for extrato in empresas['results'] if extrato['statementType'] == 'Pagamento'])
        #total_recebeminto = sum([ extrato['value'] for extrato in empresas['results'] if extrato['statementType'] == 'Recebimento'])
        #print(f'movimentações: {len(empresas["results"])}')
        #print(f'total no mês de setembro pagamento: {total_pagamento}')
        #print(f'total no mês de setembro recebimento: {total_recebeminto}')

        #contracts
        buildingId_list = [
            buildings['buildingId']
            for contract in dados_contratos['results']
            for buildings in contract['buildings']
            if re.search(r"évora", buildings['name'], re.IGNORECASE)
        ]
        buildingId = buildingId_list[0]
        print(buildingId)
        dispesas_itens = api.get_cost_estimation_itens(buildingId=buildingId)
        print(f'desepesas por item no centro de cusro:{dispesas_itens}')

    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
    except KeyError as e:
        print(f"Chave não encontrada na resposta: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()
