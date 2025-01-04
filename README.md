# DevOps CI/CD Pipelines with AI (OPCI-AI-CU)

## Descripción

Este repositorio contiene una colección de pipelines reutilizables para automatizar procesos de **Integración Continua (CI)** y **Despliegue Continuo (CD)**, optimizados con **Inteligencia Artificial (IA)**. El proyecto tiene como objetivo mejorar la eficiencia y seguridad en el desarrollo y despliegue de software, proporcionando una plataforma que los desarrolladores puedan usar para validar, construir y desplegar sus proyectos de manera rápida y segura. Además, la integración de IA permite analizar las builds, identificar fallos y sugerir mejoras.

---

## Introducción Detallada

En el desarrollo moderno de software, la automatización es una necesidad clave para garantizar calidad, velocidad y consistencia en los procesos. Este proyecto surge como una respuesta a los desafíos que enfrentan los equipos de desarrollo al implementar pipelines CI/CD:

- **Desafío técnico**: Configuración manual de pipelines para múltiples tecnologías que consume tiempo y recursos.
- **Seguridad**: Asegurar que las aplicaciones desplegadas cumplan con estándares de seguridad elevados.
- **Optimización**: Detectar y resolver ineficiencias en los procesos de build y despliegue.

Este repositorio ofrece una solución integral que combina los beneficios de la modularidad, la escalabilidad y la optimización basada en IA. La plataforma permite a los desarrolladores:

- Automatizar las tareas repetitivas del ciclo de desarrollo.
- Implementar pipelines altamente configurables con soporte para múltiples tecnologías.
- Analizar logs de manera inteligente para mejorar la calidad del código y prevenir errores futuros.

---

## Funcionalidades

- **Pipelines reutilizables**: Configuración estándar adaptable a distintos lenguajes y proyectos.
- **Automatización de CI/CD**: Desde la validación hasta el despliegue, optimizando cada etapa del proceso.
- **Análisis avanzado con IA**: Identificación de problemas, recomendaciones de optimización y generación de informes detallados.
- **Seguridad incorporada**: Integración con herramientas de análisis estático y dinámico.
- **Soporte multilingüe**: Personaliza los informes generados por la IA en varios idiomas.

---

## Requisitos

Antes de comenzar, asegúrate de tener configurados los siguientes elementos:

1. **GitHub Actions**: El motor principal para la ejecución de los pipelines.
2. **Docker**: Opcional para proyectos que requieren contenedores.
3. **Acceso a las claves necesarias**:
   - Azure Blob Storage: Para almacenamiento de logs.
   - OpenAI API Key: Para los análisis basados en IA.

---

## Guía de Configuración

### 1. Clonar el repositorio

```bash
# Usando SSH
git clone git@github.com:guillipa1993/devops-pipelines-ci-cd.git

# Usando HTTPS
git clone https://github.com/guillipa1993/devops-pipelines-ci-cd.git
```

### 2. Configurar los pipelines en tu proyecto

1. Crea un archivo `.github/workflows/ci.yml` en el repositorio de tu proyecto.
2. Define el contenido del pipeline, utilizando este repositorio como base:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches:
      - main

jobs:
  ci-pipeline:
    uses: guillipa1993/devops-pipelines-ci-cd/.github/workflows/python-ci.yml@main
    with:
      python-version: '3.9'
      report-language: 'Spanish'
    secrets:
      AZURE_STORAGE_ACCOUNT_NAME: ${{ secrets.AZURE_STORAGE_ACCOUNT_NAME }}
      AZURE_STORAGE_ACCOUNT_KEY: ${{ secrets.AZURE_STORAGE_ACCOUNT_KEY }}
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

3. Personaliza los parámetros según el lenguaje y las necesidades de tu proyecto.

---

### 3. Configurar los secretos en GitHub

1. Ve a tu repositorio en GitHub.
2. Dirígete a **Settings > Secrets and variables > Actions**.
3. Añade los siguientes secretos necesarios:
   - `AZURE_STORAGE_ACCOUNT_NAME`
   - `AZURE_STORAGE_ACCOUNT_KEY`
   - `OPENAI_API_KEY`

## Ejemplos de Configuración

Para facilitar la integración de la librería en proyectos reales, se incluyen ejemplos de archivos `ci.yml` organizados por lenguaje de programación en la carpeta [`examples`](examples/). Estos ejemplos cubren configuraciones básicas y avanzadas para diferentes tecnologías.

**Estructura de la carpeta `examples`:**

```
/examples
├── dotnet
│   └── ci-modular-todo-app.yml
├── go
│   ├── ci-cobra.yml
│   └── ci-kit.yml
├── java
│   ├── ci-junit5.yml
│   └── ci-spring-security.yml
├── nodejs
│   └── ci-create-react-app.yml
└── python
    ├── ci-requests.yml
    └── ci-scikit-learn.yml
```

**Descripción de ejemplos destacados**:

- **.NET**: Configuración para proyectos modulares, como `ci-modular-todo-app.yml`.
- **Go**: Pipelines para aplicaciones con frameworks como Cobra y Kit.
- **Java**: Ejemplos específicos con JUnit 5 y Spring Security.
- **Node.js**: Configuración para proyectos como `create-react-app`.
- **Python**: Proyectos que usan librerías como `requests` y `scikit-learn`.

Estos archivos son un punto de partida ideal para configurar pipelines en diferentes lenguajes y contextos.


---

## Notas Técnicas

### Diseño Modular
El diseño de este repositorio sigue una arquitectura modular que permite:
- Integrar diferentes lenguajes de programación.
- Adaptar los pipelines a las necesidades específicas de cada equipo.

### Integración de IA
La IA está integrada en los workflows para analizar logs, generar informes detallados y sugerir optimizaciones basadas en patrones detectados.

### Limitaciones Actuales
- **Dependencia de GitHub Actions**: El sistema no es compatible con otros motores de CI/CD como Jenkins o GitLab CI.
- **Escalabilidad del almacenamiento**: A medida que crece el volumen de logs, se incrementan los costos de almacenamiento en Azure.

### Futuras Mejoras
- Expansión del soporte a otros lenguajes y tecnologías.
- Integración con plataformas adicionales de análisis de código.
- Mayor optimización de los algoritmos de IA para una detección más precisa de errores.

---

## Licencia

Este proyecto está bajo la licencia MIT. Para más detalles, consulta el archivo [LICENSE](LICENSE).