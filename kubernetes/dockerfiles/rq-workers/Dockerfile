FROM matteocao/giotto-deep:gdeep-lab

RUN ["pip3", "install", "giotto-deep"]

CMD ["rq", "worker", "--url", "redis://redis-service", "high", "default", "low"]
