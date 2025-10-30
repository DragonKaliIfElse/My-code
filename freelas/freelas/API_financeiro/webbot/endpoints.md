### Valores de Orçamento (Orçado)

Os valores orçados podem ser encontrados principalmente nos endpoints de Orçamento da Obra, que distinguem o preço base (sem BDI) e o preço total (que geralmente inclui BDI).

| Métrica | Campo de Retorno Relevante | Endpoint(s) |
| :--- | :--- | :--- |
| **Valor Orçado da Obra S/BDI** | `baseTotalPrice` (Preço total base, sem BDI e encargos sociais) | `/building-cost-estimation-items` (Bulk Data) |
| **Valor Orçado C/BDI** | `totalPrice` (Preço total para o tipo de insumo correspondente). Este valor geralmente reflete o preço total orçado. | `/building-cost-estimations/{buildingId}/sheets/{building_unit_id}/items` |
| | `totalPrice` (Preço total calculado do insumo para o item de orçamento, com precisão de 2 casas) | `/building/resources` (Bulk Data) |

### Valores de Planejamento e Execução (Previsto e Medido)

Os valores e percentuais de planejamento (`Previsto`) e medição (`Medido` ou `Realizado`) podem ser obtidos nos módulos de Planejamento e Acompanhamento de Obras/Medições de Contratos.

| Métrica | Campo de Retorno Relevante | Endpoint(s) |
| :--- | :--- | :--- |
| **Valor Previsto** | Não há um campo de valor total previsto explícito, mas o valor pode ser inferido do planejamento das tarefas através de `quantity` (Quantidade planejada) e `totalPrice` (Preço total). | `/building-projects/{buildingId}/sheets/{buildingUnitId}/tasks` |
| **% Previsto (Caixa)** | `scheduledPercentComplete` (Percentual previsto (planejado) de execução do item de orçamento). | `/building-cost-estimation-items` (Bulk Data) |
| **Valor Medido** | `measuredTotal` (Total Medido = Total Medido de Material + Total Medido de Mão de Obra). | `/supply-contracts/measurements/all` |
| | `measuredQuantity` (Quantidade medida) | `/building-projects/{buildingId}/progress-logs/{measurementNumber}/items/{buildingUnitId}` ou `/supply-contracts/measurements/items` |
| **% Realizado** | `cumulativePercentage` (Percentual acumulado considerando a soma de todas as quantidades já registradas para este evento dividido pela quantidade planejada). | `/building-projects/progress-logs` |
| | `percentComplete` (Percentual medido do item de orçamento). | `/building-cost-estimation-items` (Bulk Data) |
| **Situação** | `status` (Situação da medição ou unidade) ou `statusApproval` (Código da situação de aprovação). | Vários endpoints, como `/supply-contracts/measurements/all` ou `/building-projects/progress-logs` ou `/units`. |

### Valores de Custos e Financeiros (Gasto, Incorrido, Contas, Estoque)

| Métrica | Campo de Retorno Relevante | Endpoint(s) |
| :--- | :--- | :--- |
| **Valor Gasto / Custo Incorrido** | Não há um campo consolidado, mas este valor é construído a partir de lançamentos em contas a pagar e movimentos de estoque, rastreados por apropriações de obra (custo/apropriação). Pode ser calculado a partir dos valores líquidos de movimentos bancários (`value`) ou parcelas pagas/apropriadas. | `/bulk-data/outcome` e `/bank-movement` |
| **Contas a Pagar** | Parcelas a pagar (`installmentId`, `netAmount`, etc.) e títulos (`billId`). | `/outcome` (Bulk Data) |
| **Estoque** | Informações atuais de insumos em estoque, incluindo `quantity` (Quantidade movimentada). | `/stock-inventories/{costCenterId}/items` e `/stock-movements` |
| **Receitas Acumuladas** | Parcelas de títulos a receber, incluindo valor líquido recebido (`netAmount`, `netReceiptValue`). | `/income` (Bulk Data) e `/accounts-receivable/receivable-bills` |
| **Despesas Acumuladas** | Lançamentos de despesas/saídas, incluindo valor líquido (`netAmount`, `correctedNetAmount`). | `/outcome` (Bulk Data) |
| **Resultado / Margem de Lucro** | Não são listados como campos diretamente retornados. Contudo, relatórios contábeis fornecem a base para o cálculo do Resultado e Margem de Lucro. | `/trial-balance` (Busca dados do balancete de verificação). |
| | Dados relacionados a custos orçados (`budgetedCost`) e variações de VGV (`vgvVariation`). | `/real-estate-map` |

### Indicadores de Desempenho (KPIs)

As fontes não listam campos com nomes exatos para os indicadores de Desempenho de Valor Agregado (EVM) como VC (Variação de Custos), IDC (Desempenho de Custos) ou IDP (Desempenho de Prazo), nem métricas específicas de custo por metro quadrado. No entanto, os componentes necessários para calcular esses KPIs estão disponíveis:

| Métrica | Campos de Base para Cálculo | Endpoint(s) de Suporte |
| :--- | :--- | :--- |
| **VC (Variação de Custos)** | Requer Custo Real (Custo Incorrido) e Custo Orçado (Previsto para o trabalho realizado), calculáveis a partir de `unitPrice` e dados de apropriação/pagamentos. | `/building-cost-estimations`, `/bulk-data/outcome` |
| **IDC (Desempenho de Custos)** | Requer valores calculados a partir dos dados de orçamento e custos reais. | N/A (Não listado explicitamente) |
| **IDP (Desempenho de Prazo)** | Requer o percentual medido (`percentComplete`) e o percentual planejado (`scheduledPercentComplete`). | `/building-cost-estimation-items` (Bulk Data) |
| **Custo/m² (Previsto/Real)** | Requer a Área (e.g., `privateArea` ou `areaQuantity`) e o Valor Orçado (`totalPrice`) ou Custo Incorrido. | `/building-cost-estimations`, `/units` |
