# 1. Resource Group
resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location
}

# 2. Storage Account
resource "azurerm_storage_account" "storage" {
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  lifecycle {
    prevent_destroy = true
  }

  tags = {
    environment = "ci-cd-logs"
  }
}

# 3. Contenedor en Blob Storage
resource "azurerm_storage_container" "logs_container" {
  name                  = "logs"
  storage_account_id    = azurerm_storage_account.storage.id
  container_access_type = "private"
}

# 7. Asignar permisos al Service Principal para acceder al Storage Account
resource "azurerm_role_assignment" "sp_role_assignment" {
  scope                = "/subscriptions/36e7912e-1fce-4ab8-8bba-4b4ef69125cf"
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = "1852e8f3-6197-4bc0-9097-d8ed29ad77b3" # ID del Service Principal ya existente
}

# 8. Key Vault para Almacenar Credenciales
resource "azurerm_key_vault" "key_vault" {
  name                       = var.key_vault_name
  location                   = azurerm_resource_group.rg.location
  resource_group_name        = azurerm_resource_group.rg.name
  sku_name                   = "standard"
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  soft_delete_retention_days = 90
  enable_rbac_authorization  = true

  tags = {
    environment = "ci-cd-logs"
  }
}

# Generar una contraseña segura aleatoria para el Service Principal
resource "random_password" "sp_password" {
  length  = 16
  special = true
}

# Datos necesarios para obtener el Tenant ID automáticamente
data "azurerm_client_config" "current" {}
