#!/bin/bash

subdomain=""
senha=""
token1="=" #base64(subdomain:senha)
token2="="
customerId=161
startDate="2025-01-01"
endDate="2025-12-01"
page=1
pageSize=100
buildingId=32
companyId=57

#resource="accounts-receivable/receivable-bills?customerId=${customerId}&startDueDate=${startDate}&endDueDate=${endDate}&page=${page}&pageSize=${pageSize}"
#resource="enterprises"
#resource="building-cost-estimations/${buildingId}/sheets"
#resource="building-cost-estimations/${buildingId}/resources"
#resource="building-cost-estimations/${buildingId}/resources/192"
#resource="supply-contracts/all?contractStartDate=${startDate}&contractEndDate=${endDate}"
resource="trial-balance?companyId=14&initialPeriod=${startDate}&finalPeriod=${endDate}"
#resource="accounts-statements?startDate=${startDate}&endDate=${endDate}"
#resource="bills?startDate=${startDate}&endDate=${endDate}"
#resource="cost-centers"
#resource="companies/13"

#resource="accountancy/accountCostCenterBalance?startDate=${startDate}&endDate=${endDate}"

url="https://api.sienge.com.br/${subdomain}/public/api/v1/${resource}?offset=0&limit=200"
#url="https://api.sienge.com.br/${subdomain}/public/api/bulk-data/v1/${resource}&offset=0&limit=200"

response=$(curl -s -X GET "$url" \
    -H "Accept: application/json" \
    -H "Authorization: Basic $token1")

echo "$url"
echo "$response" | jq '.'
#echo "$response"
