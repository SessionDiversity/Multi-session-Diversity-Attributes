Learning Diversity Attributes in Multi-Session Recommendations

## Prerequisites
Before you begin, ensure you have met the following requirements:
<!--- These are just example requirements. Add, duplicate or remove as required --->
* You have python3  installed  
* You have installed requirements  `pip install -r requirements.txt`
* You have jupyter notebook installed

## Running
To reproduce the results presented in the paper, go to ```src/``` and:
* First, run the file `embeddings.py` to import data, present in ```Data/data_files/movie_lens/10M/```, and generate a cleaned dataset which is saved in ```Data/data_files/movie_lens/10M/```. `embeddings.py` generates the embeddings of all items and saves them in ```Data/no_attr/embeddings/```. The number of the simulated and added attribute can be changed by changing the value of `num_add_attr` in the python file.
