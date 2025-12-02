FROM python:3.7-buster

ARG BUILD_ENV
ARG OVERCOOKED_BRANCH
ARG GRAPHICS

# Make sure `lsl.dll` is in the same directory as your Dockerfile or specify the correct relative path
COPY ./liblsl/bin/ /usr/local/lib/

# Set the environment variable
ENV PYLSL_LIB=/usr/local/lib/liblsl.so

# Install pylsl
RUN pip install pylsl


WORKDIR /app

# Install non-chai dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Install eventlet production server if production build
RUN if [ "$BUILD_ENV" = "production" ] ; then pip install eventlet ; fi

# Clone chai code
RUN git clone --recursive https://github.com/ahmadad0111/overcooked_ai.git --branch $OVERCOOKED_BRANCH --single-branch /overcooked_ai

# Dummy data_dir so things don't break
RUN echo "import os; DATA_DIR=os.path.abspath('.')" >> /overcooked_ai/src/human_aware_rl/data_dir.py

# Install chai dependencies
RUN pip install -e '/overcooked_ai[harl]'

RUN apt-get -y update
RUN apt-get install -y libgl1-mesa-dev

# Copy over remaining files
COPY ./static ./static
COPY ./*.py ./
COPY ./graphics/$GRAPHICS ./static/js/graphics.js
COPY ./config.json ./config.json
COPY ./db.json ./db.json



# Set environment variables that will be used by app.py
ENV HOST 0.0.0.0
ENV PORT 5000
ENV CONF_PATH config.json


# Do the thing
EXPOSE 5000
# Expose the default LSL multicast port for stream discovery
EXPOSE 16571/udp

# If your streams use other ports, expose them as well (example: 10100-10101)
EXPOSE 10100-10101/udp
EXPOSE 16572-16604/udp
EXPOSE 16572-16604/tcp
CMD ["python", "-u", "app.py"]
