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
        file_name = "logs/get_enterprises.json"
        try:
            with open(file_name, "r") as file:
                resource_estimation = json.load(file)
                return resource_estimation
        except:
            resource = f"enterprises?offset={offset}&limit={limit}"
            url = f"{self.base_url}/{resource}"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            with open(file_name, "w") as file:
                json.dump(response.json(), file, indent=2)
        return response.json()

    def get_contracts(self, offset, limit):
        file_name = "logs/get_contracts.json"
        try:
            with open(file_name, "r") as file:
                resource_estimation = json.load(file)
                return resource_estimation
        except:
            resource = f"supply-contracts/all?contractStartDate={self.start_date}&contractEndDate={self.end_date}&limit={limit}&offset={offset}"
            url = f"{self.base_url}/{resource}"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            with open(file_name, "w") as file:
                json.dump(response.json(), file, indent=2)
        return response.json()

    def get_receivable_bills(self, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
        file_name = "logs/get_receivable_bills.json"
        try:
            with open(file_name, "r") as file:
                resource_estimation = json.load(file)
                return resource_estimation
        except:
            resource = f"accounts-receivable/receivable-bills?customerId={self.customer_id}&startDueDate={self.start_date}&endDueDate={self.end_date}&page={page}&pageSize={page_size}"
            url = f"{self.base_url}/{resource}"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            with open(file_name, "w") as file:
                json.dump(response.json(), file, indent=2)
        return response.json()
 
    def get_bills(self, limit, offset):
        file_name = "logs/get_bills.json"
        try:
            with open(file_name, "r") as file:
                resource_estimation = json.load(file)
                return resource_estimation
        except:
            resource = f"bills/?startDate={self.start_date}&endDate={self.end_date}&limit={limit}&offset={offset}"
            url = f"{self.base_url}/{resource}"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            with open(file_name, "w") as file:
                json.dump(response.json(), file, indent=2)
        return response.json()

    def get_account_statement(self, limit, offset):
        file_name = "logs/get_account_statement.json"
        try:
            with open(file_name, "r") as file:
                resource_estimation = json.load(file)
                return resource_estimation
        except:
            resource = f"accounts-statements?startDate={self.start_date}&endDate={self.end_date}&offset={offset}&limit={limit}"
            url = f"{self.base_url}/{resource}"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            with open(file_name, "w") as file:
                json.dump(response.json(), file, indent=2)
        return response.json()

    def get_cost_centers(self, limit, offset): 
        file_name = "logs/log_get_cost_centers.json"
        try:
            with open(file_name, "r") as file:
                resource_estimation = json.load(file)
                return resource_estimation
        except:
            resource = f"cost-centers?offset={offset}&limit={limit}"
            url = f"{self.base_url}/{resource}"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            with open(file_name, "w") as file:
                json.dump(response.json(), file, indent=2)
        return response.json()

    def get_cost_estimation_itens(self, buildingId):
        file_name = "logs/log_get_cost_estimation_itens.json"
        try:
            with open(file_name, "r") as file:
                resource_estimation = json.load(file)
                return resource_estimation
        except:
            resource = f"building-cost-estimation-items?buildingId={buildingId}"
            url = f"{self.bulk_data_url}/{resource}"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            with open(file_name, "w") as file:
                json.dump(response.json(), file, indent=2)
        return response.json()

    def get_resources_sbdi(self, buildingId):
        file_name = "logs/log_get_resources_sbdi.json"
        try:
            with open(file_name, "r") as file:
                resource_estimation = json.load(file)
                return resource_estimation
        except:
            resource = f"building/resources?buildingId={buildingId}&startDate={self.start_date}&endDate={self.end_date}"
            url = f"{self.bulk_data_url}/{resource}"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            with open(file_name, "w") as file:
                json.dump(response.json(), file, indent=2)
        return response.json()

    def get_income(self, selectionType="I"):
        file_name = "logs/log_get_income.json"
        try:
            with open(file_name, "r") as file:
                resource_estimation = json.load(file)
                return resource_estimation
        except:
            resource = f"income?startDate={self.start_date}&endDate={self.end_date}&selectionType={selectionType}"
            url = f"{self.bulk_data_url}/{resource}"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            with open(file_name, "w") as file:
                json.dump(response.json(), file, indent=2)
        return response.json()

    def get_outcome(self, correctionIndexerId=0, selectionType="I"):
        file_name = "logs/log_get_outcome.json"
        try:
            with open(file_name, "r") as file:
                resource_estimation = json.load(file)
                return resource_estimation
        except:
            resource = f"outcome?startDate={self.start_date}&endDate={self.end_date}&selectionType={selectionType}&correctionIndexerId={correctionIndexerId}&correctionDate={self.end_date}"
            url = f"{self.bulk_data_url}/{resource}"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            with open(file_name, "w") as file:
                json.dump(response.json(), file, indent=2)
        return response.json()

    def get_building_projects(self, offset, limit, buildingUnitId=6, buildingId=161):
        file_name = "logs/get_building_projects.json"
        try:
            with open(file_name, "r") as file:
                resource_estimation = json.load(file)
                return resource_estimation
        except:
            resource = f"/building-projects/{buildingId}/sheets/{buildingUnitId}/tasks?offset={offset}&limit={limit}"
            url = f"{self.base_url}/{resource}"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            with open(file_name, "w") as file:
                json.dump(response.json(), file, indent=2)
        return response.json()

#pega os recursos sem BDI separados por categoria
def get_resources_sbdi_by_category(api, nome):
    financial_category_dict = {}
    prices = {}

    buildingId = get_buildingId(api, nome)
    print(buildingId)

    resource_estimation = api.get_resources_sbdi(buildingId=buildingId)

    financial_category_list = [ 
        resource_unit['financialCategory']
        for resource_unit in resource_estimation['data']
    ]
    financial_category_list = list(set(financial_category_list))
    for financial_category_unit in financial_category_list:
        total_price_attended = sum( [
            attended_item['value']
            for resource_unit in resource_estimation['data']
            if resource_unit['financialCategory'] == financial_category_unit
            for attended_item in resource_unit['buildingAppropriations']['attended']
            #if re.search("2025-09", attended_item['date'])
        ] )
        total_price_pending = sum( [
            attended_item['value']
            for resource_unit in resource_estimation['data']
            if resource_unit['financialCategory'] == financial_category_unit
            for attended_item in resource_unit['buildingAppropriations']['pending']
            #if re.search("2025-09", str(attended_item['date']))
        ] )
        prices['attended'] = total_price_attended
        prices['pending'] = total_price_pending
        financial_category_dict[financial_category_unit] = prices

        return financial_category_dict

#pega os preços totais atendidos e pendentes sem BDI recebendo o nome do centro de custo
def get_resources_sbdi_total(api, nome):
    buildingId = get_buildingId(api, nome)
    print(buildingId)

    resource_estimation = api.get_resources_sbdi(buildingId=buildingId)

    total_price_attended = sum( [
        attended_item['value']
        for resource_unit in resource_estimation['data']
        for attended_item in resource_unit['buildingAppropriations']['attended']
        #if re.search(ano_mes, attended_item['date'])
    ] )
    total_price_pending = sum( [
        attended_item['value']
        for resource_unit in resource_estimation['data']
        for attended_item in resource_unit['buildingAppropriations']['pending']
        #if re.search(ano_mes, str(attended_item['date']))
    ] )

    return total_price_attended, total_price_pending


#pega os preços totais separados por unidade de construção sem BDI recebendo o nome do centro de custo
def get_total_price_sbdi_sdate(api, name): 
    dict_total_price = {}
    buildingId = get_buildingId(api, name)
    print(buildingId)

    cost_estimation = api.get_cost_estimation_itens(buildingId=buildingId)

    unit_id_list = [ 
        ce_unit['buildingUnitId']
        for ce_unit in cost_estimation['data']
    ]
    unit_id_list = sorted(list(set(unit_id_list)))
    print(unit_id_list)
    for i in unit_id_list:
        total_price = sum( [
            ce_unit['baseTotalPrice']
            for ce_unit in cost_estimation['data']
            if ce_unit['buildingUnitId'] == i
        ] )
        dict_total_price[str(i)] = total_price

    return dict_total_price

def all_data(func):
    limit = 200
    empresas = func(offset=0, limit=limit)
    count = empresas['resultSetMetadata']['count']
    while limit < count:
        emp2 = func(offset=limit, limit=200)
        empresas['results'].extend(emp2['results'])
        limit += 200
    return empresas

def get_buildingId(api, name):
    dados_contratos = all_data(api.get_contracts)
    buildingId_list = [
        buildings['buildingId']
        for contract in dados_contratos['results']
        for buildings in contract['buildings']
        if re.search(name, buildings['name'], re.IGNORECASE)
    ]
    buildingId = buildingId_list[0]

    return buildingId

def main():
    api = SiengeAPI(
        subdomain="engeplan",
        password="l2dCJQaD5y21QB009C8J7nEawnpt1jmR",
        token="ZW5nZXBsYW4tYmk6bDJkQ0pRYUQ1eTIxUUIwMDlDOEo3bkVhd25wdDFqbVI",  # ou deixa None para gerar automaticamente
        start_date="2025-09-01",
        end_date="2025-10-01"
    )

    try:
        data = all_data(api.get_building_projects)
        print(json.dumps(data, indent=2))

    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
    except KeyError as e:
        print(f"Chave não encontrada na resposta: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()
