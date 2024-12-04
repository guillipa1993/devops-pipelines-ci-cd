variable "location" {
  description = "Azure region where resources will be created"
  type        = string
  default     = "East US"
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
  default     = "ci-cd-logs-rg"
}

variable "storage_account_name" {
  description = "Name of the storage account"
  type        = string
  default     = "cicdlogsstorage"
}

variable "key_vault_name" {
  description = "Name of the Key Vault"
  type        = string
  default     = "cicdlogs-kv"
}

variable "tenant_id" {
  description = "The Tenant ID for Azure Active Directory"
  type        = string
}

variable "client_id" {
  description = "The Client ID for Azure Service Principal"
  type        = string
}

variable "client_secret" {
  description = "The Client Secret for Azure Service Principal"
  type        = string
  sensitive   = true
}

variable "subscription_id" {
  description = "The Subscription ID for Azure"
  type        = string
}

variable "github_storage_access_app_id" {
  type    = string
  default = "1bb08daf-a0b9-4eb5-874d-db94d90c1d61" # Valor manual
}
