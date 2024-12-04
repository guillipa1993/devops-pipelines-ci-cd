# Ruta al directorio grafana-loki
DOCKER_COMPOSE_DIR = grafana-loki

# Nombre de los servicios de Docker
DOCKER_COMPOSE = docker-compose -f $(DOCKER_COMPOSE_DIR)/docker-compose.yml

# Ruta del directorio de Terraform
TERRAFORM_DIR = terraform

# Comando para levantar los servicios en segundo plano
up:
	@echo "Levantando Grafana, Loki y Promtail..."
	$(DOCKER_COMPOSE) up -d

# Comando para detener y eliminar los contenedores
down:
	@echo "Deteniendo y eliminando los contenedores..."
	$(DOCKER_COMPOSE) down

# Comando para ver los logs de los contenedores (Grafana, Loki, Promtail)
logs:
	@echo "Mostrando los logs de los contenedores..."
	$(DOCKER_COMPOSE) logs -f

# Limpiar los volúmenes y datos previos
clean:
	@echo "Limpiando volúmenes y datos previos..."
	rm -rf ./$(DOCKER_COMPOSE_DIR)/loki-data ./$(DOCKER_COMPOSE_DIR)/loki-wal
	mkdir -p ./$(DOCKER_COMPOSE_DIR)/loki-data/chunks ./$(DOCKER_COMPOSE_DIR)/loki-data/index ./$(DOCKER_COMPOSE_DIR)/loki-wal
	sudo chown -R 1000:1000 ./$(DOCKER_COMPOSE_DIR)/loki-data ./$(DOCKER_COMPOSE_DIR)/loki-wal
	sudo chmod -R 777 ./$(DOCKER_COMPOSE_DIR)/loki-data ./$(DOCKER_COMPOSE_DIR)/loki-wal

# Comando para reiniciar los servicios
restart: down up

# Comando para limpiar contenedores, imágenes y volúmenes no utilizados
prune:
	@echo "Limpiando recursos no utilizados (contenedores, imágenes y volúmenes)..."
	$(DOCKER_COMPOSE) down --volumes --rmi all

# Comando para verificar el estado de los contenedores
ps:
	@echo "Verificando el estado de los contenedores..."
	$(DOCKER_COMPOSE) ps

# Comando para verificar la versión de Docker Compose
version:
	@echo "Mostrando la versión de Docker Compose..."
	$(DOCKER_COMPOSE) --version

# Comando para inicializar Terraform
terraform-init:
	@echo "Inicializando Terraform..."
	cd $(TERRAFORM_DIR) && terraform init

terraform-plan:
	@echo "Realizando el Terraform Plan..."
	cd $(TERRAFORM_DIR) && terraform plan -out=tfplan.out

# Comando para aplicar la configuración de Terraform
terraform-apply:
	@echo "Aplicando configuración de Terraform..."
	cd $(TERRAFORM_DIR) && terraform apply tfplan.out

# Comando para destruir recursos de Terraform
terraform-destroy:
	@echo "Destruyendo recursos de Terraform..."
	cd $(TERRAFORM_DIR) && terraform destroy -auto-approve

# Comando para dar formato a los archivos de Terraform
terraform-format:
	@echo "Formateando los archivos de Terraform..."
	cd $(TERRAFORM_DIR) && terraform fmt -recursive

# Comando para autenticarse en Azure CLI
azure-login:
	@echo "Autenticándose en Azure CLI..."
	az login

# Comando para subir logs a Azure Blob Storage
upload-logs:
	@echo "Subiendo logs a Azure Blob Storage..."
	az storage blob upload \
		--account-name $(AZURE_STORAGE_ACCOUNT) \
		--container-name logs \
		--name grafana-logs-$(shell date "+%Y-%m-%d-%H-%M-%S").log \
		--file $(DOCKER_COMPOSE_DIR)/logs/output.log \
		--auth-mode key \
		--account-key $(AZURE_STORAGE_KEY)
