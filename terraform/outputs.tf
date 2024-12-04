output "storage_account_name" {
  description = "Name of the storage account created"
  value       = azurerm_storage_account.storage.name
}