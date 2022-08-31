Learning Diversity Attributes in Multi-Session Recommendations

## Prerequisites
Before you begin, ensure you have met the following requirements:
<!--- These are just example requirements. Add, duplicate or remove as required --->
* You have python3  installed  
* You have installed requirements  `pip install -r requirements.txt`
* You have jupyter notebook installed

## Running
To reproduce the results presented in the paper, go to ```src/``` and:
* First, run the file `embeddings.py` to import data, present in ```Data/data_files/movie_lens/10M/```, and generate a cleaned dataset which is saved in ```Data/data_files/movie_lens/10M/```. `embeddings.py` generates the embeddings of all items and saves them in ```Data/no_attr/embeddings/```. The number of simulated and added attributes can be changed by changing the value of `num_add_attr` in the python file.

* Then, run the file `reply_buffer.py` to split the user data into sessions and split these latter into training and testing sets. One can specify the number of sessions and number of items in the sessions using the arguments -num_sess and -num_items respectively. The argument -num_add_attr specifies the number of simulated attributes. Example: reply_buffer.py -num_sess 3 -num_items 5 -num_add_attr 10 generates test set of 3 sessions with 5 items in each session. The remaining sessions are used for training. The number of attributes is 15.
