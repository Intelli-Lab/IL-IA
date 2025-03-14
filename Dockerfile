ARG BUILD_FROM
FROM \$BUILD_FROM

# Installer les dépendances nécessaires
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install scikit-learn pandas flask

# Copier les fichiers nécessaires
COPY run.sh /
COPY setup.py /
RUN chmod a+x /run.sh

CMD [ "/run.sh" ]