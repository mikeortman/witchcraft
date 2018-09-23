FROM ufoym/deepo:tensorflow-py36-cu90

VOLUME ["/vol"]
ADD bootstrap.sh .
RUN chmod +x bootstrap.sh

ENTRYPOINT ["/bin/bash", "./bootstrap.sh"]
