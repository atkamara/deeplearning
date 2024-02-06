FROM python:3.11

##########################
#                        #
#        USER            #
#                        #
##########################

RUN adduser --disabled-password \
	analyst

ENV HOME /home/analyst 

#Set working directory
WORKDIR ${HOME}


#Switching to user analyst
RUN mkdir ${HOME}/notebooks
RUN chown -R analyst ${HOME} 
USER analyst


###########################
#                         # 
# Libraries               #
#                         #
###########################

SHELL ["/bin/bash","-c"]

RUN pip install --upgrade pip
RUN pip install virtualenv --progress-bar off

RUN python -m venv dlenv

RUN dlenv/bin/pip install notebook --progress-bar off

#Tesorflow

RUN dlenv/bin/pip install --upgrade pip && \
	dlenv/bin/pip install tensorflow-cpu --progress-bar off

	 	 


###########################
#                         # 
# Running process         #
#                         #
###########################

# Launch jupyter notebook on start

CMD source dlenv/bin/activate && jupyter notebook --ip 0.0.0.0 