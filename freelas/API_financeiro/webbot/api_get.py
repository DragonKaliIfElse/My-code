import requests
import base64
import json
from typing import Dict, List, Any

class SiengeAPI:
    def __init__(self, subdomain: str, password: str, token: str = ''):
        self.subdomain = subdomain
        self.password = password
        self.token = token or self._generate_token()
        self.base_url = f"https://api.sienge.com.br/{subdomain}/public/api/v1"
 
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Basic {self.token}"
        }
 
    def _generate_token(self) -> str:
        credentials = f"{self.subdomain}:{self.password}"
        return base64.b64encode(credentials.encode()).decode()

    def get_enterprises(self, offset: int = 0, limit: int = 100) -> Dict[str, Any]:
        resource = f"enterprises?offset={offset}&limit={limit}"
        url = f"{self.base_url}/{resource}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()  # Retorna como dicionário

    def get_contracts(self, offset: int = 0, limit: int = 100) -> Dict[str, Any]:
        resource = f"supply-contracts/all?contractStartDate=2025-09-01&contractEndDate=2025-10-01&limit={limit}&offset={offset}"
        url = f"{self.base_url}/{resource}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()  # Retorna como dicionário

    def get_receivable_bills(self, customer_id: int, start_date: str, end_date: str, 
                           page: int = 1, page_size: int = 100) -> Dict[str, Any]:
        resource = f"accounts-receivable/receivable-bills?customerId={customer_id}&startDueDate={start_date}&endDueDate={end_date}&page={page}&pageSize={page_size}"
        url = f"{self.base_url}/{resource}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()  # Retorna como dicionário
 
    def get_bills(self, limit, offset, start_date="2025-09-01", end_date="2025-10-01"):
        resource = f"bills/?startDate={start_date}&endDate={end_date}&limit={limit}&offset={offset}"
        url = f"{self.base_url}/{resource}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()  # Retorna como dicionário

    def get_account_statement(self, limit, offset, start_date="2025-09-01", end_date="2025-10-01"):
        resource = f"accounts-statements?startDate={start_date}&endDate={end_date}&offset={offset}&limit={limit}"
        url = f"{self.base_url}/{resource}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()  # Retorna como dicionário

    def get_cost_centers(self, limit, offset): 
        resource = f"cost-centers?offset={offset}&limit={limit}"
        url = f"{self.base_url}/{resource}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()  # Retorna como dicionário

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
        token="ZW5nZXBsYW4tYmk6bDJkQ0pRYUQ1eTIxUUIwMDlDOEo3bkVhd25wdDFqbVI"  # ou deixa None para gerar automaticamente
    )

    try:
        empresas = all_data(api.get_account_statement)

        if 'results' in empresas:
            print(json.dumps(empresas, indent=2))



        #bills (dividas e faturas) mensais
        #dividas = sum([ fat['totalInvoiceAmount'] for fat in empresas['results'] ])
        #fatura = sum([ fat['totalInvoiceAmount'] if fat["installmentsNumber"] == 1 else fat['totalInvoiceAmount']/fat['installmentsNumber'] for fat in empresas['results'] ])
        #print(f'total de faturas no mês de setembro: {fatura}')
        #print(f'valor total a pagar: {dividas}')

        #accounts-statements (extrato)
        total_pagamento = sum([ extrato['value'] for extrato in empresas['results'] if extrato['statementType'] == 'Pagamento'])
        total_recebeminto = sum([ extrato['value'] for extrato in empresas['results'] if extrato['statementType'] == 'Recebimento'])
        print(f'movimentações: {len(empresas["results"])}')
        print(f'total no mês de setembro pagamento: {total_pagamento}')
        print(f'total no mês de setembro recebimento: {total_recebeminto}')

        #contracts
        #total_recebeminto = [ contract for contract in empresas['results'] if contract['buildings']['buildingId'] == 161]
        #print(total_recebeminto)

    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
    except KeyError as e:
        print(f"Chave não encontrada na resposta: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()
