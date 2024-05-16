FROM code.ornl.gov:4567/rse/images/mantid-framework:6.8.20231027.1822-py3.10

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

RUN pip install /build/dist/subhkl-0.1.0-py3-none-any.whl
RUN rm -rf /build

CMD [ "bash" ]