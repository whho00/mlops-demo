FROM ubuntu:latest
LABEL authors="whho"

ENTRYPOINT ["top", "-b"]