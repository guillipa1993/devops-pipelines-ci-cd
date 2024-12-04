# DevOps CI/CD Pipelines with AI (OPCI-AI-CU)

## Descripción

Este repositorio contiene una colección de pipelines reutilizables para automatizar procesos de **Integración Continua (CI)** y **Despliegue Continuo (CD)**, optimizados con **Inteligencia Artificial (IA)**. El proyecto tiene como objetivo mejorar la eficiencia y seguridad en el desarrollo y despliegue de software, proporcionando una plataforma que los desarrolladores puedan usar para validar, construir y desplegar sus proyectos de manera rápida y segura. Además, la integración de IA permite analizar las builds, identificar fallos y sugerir mejoras.

Ahora incluye soporte para **observabilidad y monitoreo** con Grafana Loki, facilitando el análisis de logs y la depuración en tiempo real.

## Funcionalidades

- **Pipelines reutilizables**: Puedes aplicar estos pipelines en diferentes proyectos seleccionando la tecnología deseada.
- **Automatización de CI/CD**: Validación, construcción y despliegue automatizados para proyectos con diferentes tecnologías (Node.js, Java, Python, etc.).
- **Seguridad**: Integración de herramientas de análisis de código y control de calidad.
- **Optimización con IA**: Análisis de los resultados de las builds para mejorar la eficiencia, identificar errores recurrentes y sugerir optimizaciones.
- **Observabilidad con Grafana Loki**: Centralización y análisis de logs generados por los pipelines y las aplicaciones.

## Requisitos

Para utilizar estos pipelines y la funcionalidad de monitoreo, asegúrate de cumplir con los siguientes requisitos:

- **GitHub Actions**: Los pipelines están diseñados para ejecutarse en GitHub Actions.
- **Docker**: Para algunas tecnologías y para el stack de monitoreo.
- **Docker Compose**: Para desplegar Grafana, Loki y Promtail.
- **Acceso a un repositorio** donde se pueda ejecutar el pipeline.

## Uso

### Clonar el repositorio

Puedes clonar este repositorio a tu entorno local usando SSH o HTTPS:

```bash
# Usando SSH
git clone git@github.com:guillipa1993/devops-pipelines-ci-cd.git

# Usando HTTPS
git clone https://github.com/guillipa1993/devops-pipelines-ci-cd.git
```

### Configuración del pipeline

1. **Configura el pipeline en tu proyecto**:
   - En el repositorio donde deseas usar el pipeline, crea un archivo `.github/workflows/ci.yml`.
   - Dentro del archivo YAML, invoca los pipelines reutilizables de este repositorio.

   Ejemplo de archivo `ci.yml` en tu proyecto:
   ```yaml
   name: CI/CD Pipeline

   on:
     push:
       branches:
         - main

   jobs:
     ci-pipeline:
       uses: guillipa1993/devops-pipelines-ci-cd/.github/workflows/ci.yml@main
       with:
         node-version: '14'  # Especifica la versión de Node.js, o adapta según la tecnología
       secrets: inherit
   ```

2. **Configuración específica del proyecto**:
   - Puedes pasar parámetros como la versión de la tecnología utilizada (por ejemplo, `node-version`, `java-version`), entre otros.

### Configuración de monitoreo con Grafana Loki

1. **Inicia el stack de observabilidad**:
   - En el directorio `grafana-loki`, usa Docker Compose para desplegar Grafana, Loki y Promtail:
     ```bash
     cd grafana-loki
     docker-compose up -d
     ```
   - Esto desplegará los siguientes servicios:
     - **Grafana** en `http://localhost:3000`
     - **Loki** como backend de logs.
     - **Promtail** como agente recolector de logs.

2. **Configura Grafana**:
   - Accede a la interfaz de Grafana en `http://localhost:3000` (usuario: `admin`, contraseña: `admin`).
   - Agrega Loki como una fuente de datos:
     - Ve a **Configuration > Data Sources > Add data source**.
     - Selecciona **Loki** e ingresa la URL: `http://loki:3100`.

3. **Monitoreo en tiempo real**:
   - Visualiza los logs recolectados desde los pipelines y aplicaciones en el panel de Grafana.
   - Usa los dashboards preconfigurados o crea uno personalizado para tus necesidades.

### Integración de IA

La Inteligencia Artificial integrada en estos pipelines analiza los registros (logs) de las builds, detectando patrones y errores recurrentes. Esto permitirá:

- **Identificar problemas automáticamente** y prevenir fallos futuros.
- **Recomendar optimizaciones** en las builds basadas en el análisis de los datos históricos.

## Cómo contribuir

Si deseas contribuir a este proyecto, sigue estos pasos:

1. Haz un **fork** de este repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza tus cambios y **haz commit** (`git commit -m 'Añadir nueva funcionalidad'`).
4. Haz un **push** a la rama (`git push origin feature/nueva-funcionalidad`).
5. Abre una **pull request** para revisar tus cambios.

## Licencia

Este proyecto está bajo la licencia MIT. Para más detalles, consulta el archivo [LICENSE](LICENSE).

### Explicación del contenido:
- **Descripción general**: Explica los objetivos del repositorio y sus características principales, incluyendo observabilidad con Grafana Loki.
- **Funcionalidades**: Detalla lo que ofrece el repositorio, como los pipelines reutilizables, integración de IA y monitoreo.
- **Requisitos**: Lista las herramientas necesarias, como Docker, Docker Compose y GitHub Actions.
- **Instrucciones de uso**: Cómo clonar el repositorio, configurar los pipelines y habilitar el monitoreo.
- **Integración de IA**: Describe cómo se analiza y optimiza el proceso de CI/CD mediante IA.
- **Guía para contribuir**: Explica los pasos para colaborar con el proyecto.