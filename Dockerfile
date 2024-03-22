FROM python:3.10-slim

WORKDIR /code

ARG USERNAME=starling
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update && \
    apt-get install -y build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && chown -R ${USER_UID}:${USER_GID} /code

# Install Poetry
# https://github.com/python-poetry/poetry/issues/6397#issuecomment-1236327500

ENV POETRY_HOME=/opt/poetry

# install poetry into its own venv
RUN python3 -m venv $POETRY_HOME && \
    $POETRY_HOME/bin/pip install poetry==1.7.1

ENV VIRTUAL_ENV=/poetry-env \
    PATH="/poetry-env/bin:$POETRY_HOME/bin:$PATH"

RUN python3 -m venv $VIRTUAL_ENV && \
    chown -R $USER_UID:$USER_GID $POETRY_HOME /poetry-env

USER $USERNAME

COPY . .

RUN poetry install --with docs,dev
