FROM code.ornl.gov:4567/rse/images/mantid-framework:6.8.20231027.1822-py3.10 as build

ENV PATH="/root/.local/bin:$PATH"

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

FROM code.ornl.gov:4567/rse/images/mantid-framework:6.8.20231027.1822-py3.10 as tool

COPY --from=build dist/subhkl-0.1.0.dev9+d202405011731-py3-none-any.whl subhkl.whl
RUN pip install subhkl.whl

CMD [ "bash" ]