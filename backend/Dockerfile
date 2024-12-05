FROM python:3.10.10-slim
ENV PYTHONUNBUFFERED=true
WORKDIR /app

ENV POETRY_HOME=/opt/poetry
# ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN apt-get update && \
    apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - 

RUN poetry config virtualenvs.create false
# Copy poetry.lock* in case it doesn't exist in the repo
COPY ./pyproject.toml ./poetry.lock* /app/

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi"

COPY . /app

ARG BACKEND_ENV=production

ENV PYTHON_APP_ENV=${BACKEND_ENV}

ENV PYTHONPATH=/app
