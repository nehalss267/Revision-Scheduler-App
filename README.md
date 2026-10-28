# Backend Project

## Overview

This is a complete Django REST API project built with Django REST Framework (DRF) and includes interactive API documentation using Swagger/OpenAPI (drf-yasg). The project provides a clean, scalable backend structure with example API endpoints for health checks and user management.

## Features

- **Django 5.2** - Modern Python web framework
- **Django REST Framework** - Powerful toolkit for building Web APIs
- **Swagger/OpenAPI Documentation** - Interactive API docs with Swagger UI and ReDoc
- **Environment Configuration** - Secure configuration using .env files
- **RESTful Architecture** - Clean API design following REST principles
- **CORS Ready** - Configurable for cross-origin requests
- **Production Ready** - Includes deployment best practices

## Tech Stack

- Python 3.11+
- Django 5.2.18
- Django REST Framework 3.18.3
- drf-yasg 1.21.18
- SQLite (default) / PostgreSQL ready

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.11 or higher
- pip (Python package manager)
- Virtual environment tool (venv)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/nehalss267/NehalSolanki_BackendFinalAssesement.git
```


### 2. Create and Activate Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


### 4. Configure Environment Variables

Copy the example environment file and update it with your settings:

```bash
cp .env.example .env
```

Edit .env file with your configuration:
```env
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1
```


### 5. Run Database Migrations

```bash
python manage.py migrate
```

### 6. Create Superuser (Optional)

To access the Django admin panel:
```bash
python manage.py createsuperuser
```


### 7. Run Development Server

```bash
python manage.py runserver
```


The server will start at http://localhost:8000

## API Documentation

Once the server is running, access the interactive API documentation:

| Documentation | URL | Description |
|---------------|-----|-------------|
| Swagger UI | [http://localhost:8000/swagger/](http://localhost:8000/swagger/) | Interactive API testing interface |
| ReDoc | [http://localhost:8000/redoc/](http://localhost:8000/redoc/) | Clean, responsive API documentation |
| API Root | [http://localhost:8000/api/](http://localhost:8000/api/) | Browseable API root |

## Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/health/ | GET | Check API health status |
| /api/health/ping/ | GET | Simple ping test endpoint |
| /api/users/ | GET | List all registered users |
| /api/users/{id}/ | GET | Retrieve specific user details |

## Admin Panel

Access the Django admin interface at [http://localhost:8000/admin/](http://localhost:8000/admin/) to manage users and data.

## Project Structure

`
backendProject/
├── api/                # API application
│   ├── serializers.py  # Data serializers
│   ├── views.py        # API views/viewsets
│   ├── urls.py         # API routes
│   └── ...
├── core/               # Project configuration
│   ├── settings.py     # Django settings
│   ├── urls.py         # Main URL configuration
│   └── ...
├── manage.py           # Django management script
├── requirements.txt    # Project dependencies
├── .env.example        # Environment variables template
└── README.md           # Project documentation
`

## Running Tests

```bash
python manage.py test
`

## Deployment Notes

For production deployment:
1. Set DEBUG=False in .env
2. Update SECRET_KEY with a secure random key
3. Configure ALLOWED_HOSTS with your domain
4. Use a production database (PostgreSQL recommended)
5. Collect static files: python manage.py collectstatic
6. Use a WSGI server like Gunicorn

## License

This project is for educational purposes.
=======

