from pydantic import BaseModel
import os
import sys
import logging
# current_dir = os.getcwd()

# parent_dir_two_levels_back = os.path.dirname(os.path.realpath(current_dir))
# print(parent_dir_two_levels_back)
# if parent_dir_two_levels_back not in sys.path:
#     sys.path.append(parent_dir_two_levels_back)
log_format = '%(asctime)s:%(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)


class QuestionAnswerRequest(BaseModel):
    question: str
    api_key: str
