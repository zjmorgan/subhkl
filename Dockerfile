FROM continuumio/miniconda as build

ENV PATH="/root/.local/bin:$PATH"

ADD environment.yml environment.yml
RUN conda env create --file=environment.yml

RUN apt update 
RUN apt install -y curl make 
RUN rm -rf /var/lib/apt/lists/* 
RUN curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python -

RUN mkdir /build
WORKDIR /build
ADD pyproject.toml /build/pyproject.toml
ADD pdm.lock /build/pdm.lock
RUN pdm install --frozen-lockfile --no-self

ADD . /build

RUN pdm build

FROM continuumio/miniconda as tool

ADD environment.yml environment.yml
RUN conda env create --file=environment.yml
COPY --from=build dist/subhkl-0.1.0.dev9+d202405011731-py3-none-any.whl subhkl.whl
RUN pip install subhkl.whl

CMD [ "bash" ]