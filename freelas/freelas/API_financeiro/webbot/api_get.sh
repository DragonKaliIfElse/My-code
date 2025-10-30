#!/bin/bash

subdomain="engeplan"
senha="l2dCJQaD5y21QB009C8J7nEawnpt1jmR"
token1="ZW5nZXBsYW4tYmk6bDJkQ0pRYUQ1eTIxUUIwMDlDOEo3bkVhd25wdDFqbVI=" #base64(subdomain:senha)
token2="ZW5nZXBsYW4tY2xhdWRpbzpsMFVoZG9sNGl6YUdvS0g5UXFBNjdPdFlUSWYxZDI4OAo="
customerId=161
startDate="2025-01-01"
endDate="2025-12-01"
page=1
pageSize=100
buildingId=161
companyId=57

#normal
#resource="accounts-receivable/receivable-bills?customerId=${customerId}&startDueDate=${startDate}&endDueDate=${endDate}&page=${page}&pageSize=${pageSize}&"
#resource="enterprises?"
#resource="building-cost-estimations/${buildingId}/sheets?"
#resource="building-cost-estimations/${buildingId}/resources?"
#resource="building-cost-estimations/${buildingId}/resources/192?"
#resource="supply-contracts/all?contractStartDate=${startDate}&contractEndDate=${endDate}&"
#resource="trial-balance?companyId=14&initialPeriod=${startDate}&finalPeriod=${endDate}&"
#resource="accounts-statements?startDate=${startDate}&endDate=${endDate}&"
#resource="bills?startDate=${startDate}&endDate=${endDate}&"
#resource="cost-centers?"
#resource="companies/13?"
#resource="sales-contracts/?"

#bulk-data
#resource="accountancy/accountCostCenterBalance?startDate=${startDate}&endDate=${endDate}"
#resource="outcome?startDate=${startDate}&endDate=${endDate}&selectionType=D&"
resource="building-cost-estimation-items?buildingId=${buildingId}"

#url="https://api.sienge.com.br/${subdomain}/public/api/v1/${resource}offset=0&limit=200"
url="https://api.sienge.com.br/${subdomain}/public/api/bulk-data/v1/${resource}"

response=$(curl -s -X GET "$url" \
    -H "Accept: application/json" \
    -H "Authorization: Basic $token1")

echo "$url"
echo "$response" | jq '.'
#echo "$response"
