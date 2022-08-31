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

* Running the file `reply_buffer_transfer.py` is similar to running `reply_buffer.py` with one difference. This file classifies the users into 3 classes based on their overall diversity (Section V-E in the paper) and generates train and test sets for each class used to perform transfer learning.

* If one wants to perform parameters tuning, she/he runs the file `smorl_parameters.py`. The tuned parameters as well as their possible values are the ones mentioned in the paper. The models are saved in ```Data/no_attr/models/parameters_study/```. The script uses train sets generated by `reply_buffer.py`. The arguments are -num_sess, -num_items, and -num_add_attr.

* `smorl_learning.py` is the file used to train our RL model using the train set generated before for \[0,15,25,35,100\] number of attributes using the best parameters mentioned in the paper. Before running this file, make sure that all required embeddings are generated as well as the training and testing sets. The trained models are saved in ```Data/no_attr/models/```. The arguments are -num_sess and -num_items.
