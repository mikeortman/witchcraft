FROM ufoym/deepo:tensorflow-py36-cpu

VOLUME ["/vol"]
ADD bootstrap.sh .
RUN chmod +x bootstrap.sh

ENTRYPOINT ["/bin/bash", "./bootstrap.sh"]
