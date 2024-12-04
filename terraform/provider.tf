terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "4.12.0" # Versión específica que estás utilizando actualmente
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "3.0.2" # Versión específica que estás utilizando actualmente
    }
  }

  required_version = ">= 0.14.9"
}

# Proveedor de Azure Resource Manager (ARM)
provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
  client_id       = var.client_id
  client_secret   = var.client_secret
  tenant_id       = var.tenant_id
}

# Proveedor de Azure Active Directory
provider "azuread" {
  tenant_id     = var.tenant_id
  client_id     = var.client_id
  client_secret = var.client_secret
}
