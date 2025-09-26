from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
from flashrag.config import Config

config_dict = {'data_dir': 'scratch/'}
my_config = Config(
    # Create my own config later.
    config_file_path='flashrag/config/basic_config.yaml',
    config_dict=config_dict
)

all_split = get_dataset(my_config)
test_data = all_split['test']
print(test_data)

pipeline = SequentialPipeline(my_config)

output_dataset = pipeline.run(test_data, do_eval=True)